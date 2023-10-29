from torch.utils.tensorboard import SummaryWriter
import os
import random
import torch
import sys
import torch.nn as nn
import itertools
from itertools import zip_longest
from torch.autograd import Variable

from Generator import U_Net
from Discriminator import MultiScaleDiscriminator_Scale4_1, MultiScaleDiscriminator_Scale4_4, MultiScaleDiscriminator_Scale4_8, MultiScaleDiscriminator_Scale4_31_22, MultiScaleDiscriminator_Scale4_62_45
from utils import dumpPickle, cal_PESQ, cal_PESQ_fake, my_makedir
from losses import GANLoss

class Trainer(object):
    def __init__(self, clean_loader, train_loader, val_loader, args):
        # data loader
        self.clean_loader = clean_loader
        self.tarin_loader = train_loader
        self.val_loader = val_loader

        # Model configuration.
        self.args = args
        self.device = torch.device(f'cuda:{args.cuda_num}' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        print(torch.__version__)

        # create directories if not exist.
        my_makedir(os.path.join(args.save_path, "train/log"))
        my_makedir(os.path.join(args.save_path, "train/model"))
        my_makedir(os.path.join(args.save_path, "val/log"))

        # Directories.
        self.log_path = os.path.join(args.save_path, "train/log")
        self.model_path = os.path.join(args.save_path, "train/model")
        self.val_log_path = os.path.join(args.save_path, "val/log")

        # Build the model and tensorboard.
        self.build_model()
        if self.args.use_tensorboard:
            self.build_tensorboard()

    def train(self):
        """ Train CycleGAN ."""
        # 損失関数
        self.criterionGAN = GANLoss(self.args.adv_loss_type, tensor=torch.cuda.FloatTensor)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = self.identity_loss(self.args.idt_loss_type)

        # 過去データ分のメモリ確保
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        # start training
        losses_D = []
        losses_G = []
        losses_G_Idt = []
        losses_G_GAN = []
        losses_G_Cycle = []

        val_PESQ_fake_data = []
        print("======================================= start training =======================================")
        #ドメインA :  品質の悪い音声信号
        #ドメインB :  クリーンな音声信号
        for epoch in range(1, self.args.total_epochs+1):
            print(f"[Epoch: {epoch}]")
            self.G_A2B.train()
            self.G_B2A.train()
            self.D_A.train()
            self.D_B.train()
            running_loss_D = 0.0
            running_loss_G = 0.0
            running_loss_G_GAN = 0.0
            running_loss_G_Idt = 0.0
            running_loss_G_Cycle = 0.0
            for count, (d_train, d_clean) in enumerate(zip_longest(self.tarin_loader, self.clean_loader), 1):
                # モデルの入力
                real_A = Variable(d_train["ase_src"])
                real_B = Variable(d_clean["clean_src"])

                ##### 生成器A2B、B2Aの処理 #####
                self.optimizer_G.zero_grad()

                # 同一性損失の計算（Identity loss)
                if epoch <= self.args.idt_reset_epoch:
                    # G_A2B(B)はBと一致
                    same_B = self.G_A2B(real_B)
                    loss_identity_B = self.criterionIdt(same_B, torch.abs(real_B)) * self.args.lambda_idt
                    # G_B2A(A)はAと一致
                    same_A = self.G_B2A(real_A)
                    loss_identity_A = self.criterionIdt(same_A, torch.abs(real_A)) * self.args.lambda_idt

                # 敵対的損失（GAN loss）
                fake_B = self.G_A2B(real_A)
                pred_fake = self.D_B(fake_B)
                pred_real = self.D_B(real_B)
                loss_GAN_A2B = self.criterionGAN(pred_fake, pred_real, None, for_discriminator=False) * self.args.lambda_adv

                fake_A = self.G_B2A(real_B)
                pred_fake = self.D_A(fake_A)
                pred_real = self.D_B(real_A)
                loss_GAN_B2A = self.criterionGAN(pred_fake, pred_real, None, for_discriminator=False) * self.args.lambda_adv

                # サイクル一貫性損失（Cycle-consistency loss）
                recovered_A = self.G_B2A(fake_B)
                loss_cycle_ABA = self.criterionCycle(recovered_A, torch.abs(real_A)) * self.args.lambda_cycle
                recovered_B = self.G_A2B(fake_A)
                loss_cycle_BAB = self.criterionCycle(recovered_B, torch.abs(real_B)) * self.args.lambda_cycle

                # 生成器の合計損失関数（Total loss）
                if epoch <= self.args.idt_reset_epoch:
                    loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                else:
                    loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()
                self.optimizer_G.step()

                ##### ドメインAの識別器 #####
                self.optimizer_D_A.zero_grad()

                # ドメインAの本物信号,生成信号の識別結果
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.D_A(fake_A.detach())
                pred_real = self.D_A(real_A)
                loss_D_A = self.criterionGAN(pred_fake, pred_real, None, for_discriminator=True) * self.args.lambda_adv

                loss_D_A.backward()
                self.optimizer_D_A.step()

                ##### ドメインBの識別器 #####
                self.optimizer_D_B.zero_grad()

                # ドメインBの本物信号,生成信号の識別結果
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.D_B(fake_B.detach())
                pred_real = self.D_B(real_B)
                loss_D_B = self.criterionGAN(pred_fake, pred_real, None, for_discriminator=True) * self.args.lambda_adv

                loss_D_B.backward()
                self.optimizer_D_B.step()

                running_loss_D += (loss_D_A + loss_D_B)
                running_loss_G += loss_G
                running_loss_G_Cycle += (loss_cycle_ABA + loss_cycle_BAB)
                running_loss_G_GAN += (loss_GAN_A2B + loss_GAN_B2A)
                if epoch <= self.args.idt_reset_epoch:
                    running_loss_G_Idt += (loss_identity_A + loss_identity_B)

            #Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()

            running_loss_D /= count
            running_loss_G /= count
            running_loss_G_GAN /= count
            running_loss_G_Cycle /= count
            losses_D.append(running_loss_D.cpu().item())
            losses_G.append(running_loss_G.cpu().item())
            losses_G_GAN.append(running_loss_G_GAN.cpu().item())
            losses_G_Cycle.append(running_loss_G_Cycle.cpu().item())

            dumpPickle(os.path.join(self.log_path, "losses_D.pickle"), losses_D)
            dumpPickle(os.path.join(self.log_path, "losses_G.pickle"), losses_G)
            dumpPickle(os.path.join(self.log_path, "losses_G_GAN.pickle"), losses_G_GAN)
            dumpPickle(os.path.join(self.log_path, "losses_G_Cycle.pickle"), losses_G_Cycle)

            if epoch <= self.args.idt_reset_epoch:
                losses_G_Idt.append(running_loss_G_Idt.cpu().item())
                running_loss_G_Idt /= count
                dumpPickle(os.path.join(self.log_path, "losses_G_Idt.pickle"), losses_G_Idt)

            if epoch <= self.args.idt_reset_epoch:
                print('epoch: {}, D loss: {}, G loss: {}, G loss Idt:{}, G loss Cycle:{}, G loss GAN:{} alpha: {}_{}_{}'.format(epoch, running_loss_D.cpu().item(), running_loss_G.cpu().item(), running_loss_G_Idt.cpu().item(), running_loss_G_Cycle.cpu().item(), running_loss_G_GAN.cpu().item(), self.args.lambda_idt, self.args.lambda_cycle, self.args.lambda_adv))
            else:
                print('epoch: {}, D loss: {}, G loss: {}, G loss Idt:{}, G loss Cycle:{}, G loss GAN:{} alpha: {}_{}_{}'.format(epoch, running_loss_D.cpu().item(), running_loss_G.cpu().item(), "None", running_loss_G_Cycle.cpu().item(), running_loss_G_GAN.cpu().item(), "None", self.args.lambda_cycle, self.args.lambda_adv))

            if self.args.use_tensorboard:
                self.writer.add_scalar("Loss_G", running_loss_G, epoch)
                self.writer.add_scalar("Loss_D", running_loss_D, epoch)
                self.writer.add_scalar("Loss_G_Idt", running_loss_G_Idt, epoch)
                self.writer.add_scalar("Loss_G_Cycle", running_loss_G_Cycle, epoch)
                self.writer.add_scalar("Loss_G_GAN", running_loss_G_GAN, epoch)

            #------------------------Validation---------------------
            if epoch >= self.args.num_epochs_start_val:
                self.G_A2B.eval()

                print("(validation)")
                val_PESQ_fake = 0
                val_PESQ_ase = 0

                for val_count, d in enumerate(self.val_loader, 1):
                    val_clean_src = d["clean_src"].to(self.device, non_blocking=True)
                    val_ase_src = d["ase_src"].to(self.device, non_blocking=True)
                    # 生成器に入力
                    val_fake_src = self.G_A2B(val_ase_src)

                    val_PESQ_fake += cal_PESQ_fake(val_clean_src, val_ase_src, val_fake_src, self.device)

                    if epoch==1:
                        val_PESQ_ase += cal_PESQ(val_clean_src, val_ase_src)

                val_PESQ_average_fake = val_PESQ_fake/len(self.val_loader)
                if epoch==1:
                    val_PESQ_average_ase = val_PESQ_ase/len(self.val_loader)
                    best_val_PESQ_fake = 0

                val_PESQ_fake_data.append(val_PESQ_average_fake)
                if best_val_PESQ_fake <= val_PESQ_average_fake:
                    best_val_PESQ_fake = val_PESQ_average_fake
                    print("******************good_model***********************")
                    torch.save(self.G_A2B.state_dict(), os.path.join(self.model_path, f"G_modelA2B_{epoch}.pth"))
                    torch.save(self.G_B2A.state_dict(), os.path.join(self.model_path, f"G_modelB2A_{epoch}.pth"))

                print("-----------------results----------------")
                print(f"val_PESQ_ase:  {val_PESQ_average_ase}       val_PESQ_fake:  {val_PESQ_average_fake}")

                if self.args.use_tensorboard:
                    self.writer_val.add_scalar("val_PESQ", val_PESQ_average_fake, epoch)

                dumpPickle(os.path.join(self.val_log_path, f"PESQ_fake.pickle"), val_PESQ_fake_data)

    ###define some functions###
    def build_model(self):
        """Create a generator and a discriminator."""
        # Generator
        if self.args.Generator == "U_Net":
            self.G_A2B = U_Net().to(self.device)
            self.G_B2A = U_Net().to(self.device)
        else:
            raise ValueError("Generator is ....... /trainer.py")

        # Discriminator
        if self.args.Discriminator == "MultiScaleDiscriminator_Scale4_1":
            self.D_A = MultiScaleDiscriminator_Scale4_1(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
            self.D_B = MultiScaleDiscriminator_Scale4_1(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
        elif self.args.Discriminator == "MultiScaleDiscriminator_Scale4_4":
            self.D_A = MultiScaleDiscriminator_Scale4_4(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
            self.D_B = MultiScaleDiscriminator_Scale4_4(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
        elif self.args.Discriminator == "MultiScaleDiscriminator_Scale4_8":
            self.D_A = MultiScaleDiscriminator_Scale4_8(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
            self.D_B = MultiScaleDiscriminator_Scale4_8(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
        elif self.args.Discriminator == "MultiScaleDiscriminator_Scale4_31_22":
            self.D_A = MultiScaleDiscriminator_Scale4_31_22(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
            self.D_B = MultiScaleDiscriminator_Scale4_31_22(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
        elif self.args.Discriminator == "MultiScaleDiscriminator_Scale4_62_45":
            self.D_A = MultiScaleDiscriminator_Scale4_62_45(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
            self.D_B = MultiScaleDiscriminator_Scale4_62_45(self.args.d_conv_dim, self.args.d_norm_fun, self.args.d_act_fun, self.args.d_use_sn, self.args.adv_loss_type).to(self.device)
        else:
            raise ValueError("Discriminator is ....... /trainer.py")
        print("=== Models have been created ===")

        # print network
        if self.args.is_print_network:
            self.print_network(self.G_A2B, 'G_A2B')
            self.print_network(self.G_B2A, 'G_B2A')
            self.print_network(self.D_A, 'D_A')
            self.print_network(self.D_A, 'D_B')

        # optimizer
        if self.args.optimizer_type == 'adam':
            # Adam optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
                                lr=self.args.g_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=self.args.d_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
            self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=self.args.d_lr, betas=[self.args.beta1, self.args.beta2], weight_decay=0.0001)
        elif self.args.optimizer_type == 'rmsprop':
            # RMSprop optimizer
            self.optimizer_G = torch.optim.RMSprop(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                lr=self.args.g_lr, betas=[self.args.beta1, self.args.beta2])
            self.optimizer_D_A = torch.optim.RMSprop(self.D_A.parameters(), lr=self.args.d_lr, betas=[self.args.beta1, self.args.beta2])
            self.optimizer_D_B = torch.optim.RMSprop(self.D_B.parameters(), lr=self.args.d_lr, betas=[self.args.beta1, self.args.beta2])
        else:
            raise NotImplementedError("=== Optimizer [{}] is not found ===".format(self.args.optimizer_type))

        # learning rate decay
        if self.args.lr_decay:
            def lambda_rule(epoch):
                return 1.0 - max(0, epoch + 1 - self.args.lr_num_epochs_decay) / self.args.lr_decay_ratio
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
            self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=lambda_rule)
            self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=lambda_rule)
            print("=== Set learning rate decay policy for Generator(G) and Discriminator(D) ===")

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("=== The number of parameters of the above model [{}] is [{}] or [{:>.4f}M] ===".format(name, num_params, num_params / 1e6))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_path, "tensorboard"))
        self.writer_val = SummaryWriter(log_dir=os.path.join(self.val_log_path, "tensorboard"))

    def identity_loss(self, idt_loss_type):
        if idt_loss_type == 'l1':
            criterion = nn.L1Loss()
            return criterion
        elif idt_loss_type == 'smoothl1':
            criterion = nn.SmoothL1Loss()
            return criterion
        elif idt_loss_type == 'l2':
            criterion = nn.MSELoss()
            return criterion
        else:
            raise NotImplementedError("=== Identity loss type [{}] is not implemented. ===".format(self.args.idt_loss_type))

# 過去の生成データ(50iter分)を保持しておく
class ReplayBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            #
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
