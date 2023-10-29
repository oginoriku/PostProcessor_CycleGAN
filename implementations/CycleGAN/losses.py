import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio.transforms as T

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        elif gan_mode == 'rahinge':
            pass
        elif gan_mode == 'rals':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        if self.gan_mode == 'original': # cross entropy loss
            if for_real:
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(real_preds, target_tensor)
                return loss
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(fake_preds, target_tensor)
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'ls':
            if for_real:
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                return F.mse_loss(real_preds, target_tensor)
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                return F.mse_loss(fake_preds, target_tensor)
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'hinge':
            if for_real:
                if for_discriminator:
                    if target_is_real:
                        minval = torch.min(real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(real_preds)
                return loss
            elif for_fake:
                if for_discriminator:
                    if target_is_real:
                        minval = torch.min(fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(fake_preds)
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'rahinge':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff))
                return loss
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))
                return loss
        elif self.gan_mode == 'rals':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds) #-1.0 〜 1.0
                loss = torch.mean((r_f_diff - 1) ** 2) + torch.mean((f_r_diff + 1) ** 2)
                return loss
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean((r_f_diff + 1) ** 2) + torch.mean((f_r_diff - 1) ** 2)
                return loss
        else:
            # wgan
            if for_real:
                if target_is_real:
                    return -real_preds.mean()
                else:
                    return real_preds.mean()
            elif for_fake:
                if target_is_real:
                    return -fake_preds.mean()
                else:
                    return fake_preds.mean()
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")

    def __call__(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        ## computing loss is a bit complicated because |input| may not be
        ## a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(real_preds, list):
            loss = 0
            for (pred_real_i, pred_fake_i) in zip(real_preds, fake_preds):
                if isinstance(pred_real_i, list):
                    pred_real_i = pred_real_i[-1]
                if isinstance(pred_fake_i, list):
                    pred_fake_i = pred_fake_i[-1]

                loss_tensor = self.loss(pred_real_i, pred_fake_i, target_is_real, for_real, for_fake, for_discriminator)

                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss
        else:
            return self.loss(real_preds, target_is_real, for_discriminator)

