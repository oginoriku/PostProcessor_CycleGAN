import argparse
from utils import str2bool


def get_config():
    parser = argparse.ArgumentParser()

    # Data configuration
    parser.add_argument('--f_size', type=int, default=257, help='f_size')
    parser.add_argument('--t_size', type=int, default=189, help='t_size')
    parser.add_argument('--fs', type=int, default=16000, help='fs')
    parser.add_argument('--nperseg', type=int, default=512, help='nperseg')
    parser.add_argument('--noverlap', type=int, default=256, help='noverlap')

    # Model configuration.
    parser.add_argument('--mode', type=str, default='train', help='train')
    parser.add_argument("--Generator", default='U_Net', choices=["U_Net"], type=str)
    parser.add_argument("--Discriminator", default='MultiScaleDiscriminator_Scale4_62_45', choices=["MultiScaleDiscriminator_Scale4_1", "MultiScaleDiscriminator_Scale4_4", "MultiScaleDiscriminator_Scale4_8", "MultiScaleDiscriminator_Scale4_31_22", "MultiScaleDiscriminator_Scale4_62_45"], type=str)
    parser.add_argument('--g_conv_dim', type=int, default=16, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_use_sn', type=str2bool, default=False, help='whether use spectral normalization in G')
    parser.add_argument('--d_use_sn', type=str2bool, default=True, help='whether use spectral normalization in D')
    parser.add_argument('--g_act_fun', type=str, default='LeakyReLU', help='LeakyReLU|ReLU|Swish|SELU|none')
    parser.add_argument('--d_act_fun', type=str, default='LeakyReLU', help='LeakyReLU|ReLU|Swish|SELU|none')
    parser.add_argument('--g_norm_fun', type=str, default='none', help='BatchNorm|InstanceNorm|none')
    parser.add_argument('--d_norm_fun', type=str, default='none', help='BatchNorm|InstanceNorm|none')

    # Training configuration.
    parser.add_argument('--total_epochs', type=int, default=300, help='total epochs to update the generator')
    parser.add_argument('--batch_size', type=int, default=16, help='mini batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='mini batch size')
    parser.add_argument('--clean_data_num', type=int, default=10240, help='clean data number')
    parser.add_argument('--train_data_num', type=int, default=10240, help='train data number')
    parser.add_argument('--val_data_num', type=int, default=600, help='val data number')
    parser.add_argument('--num_workers', type=int, default=2, help='subprocesses to use for data loading')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--lr_decay', type=str2bool, default=True, help='setup learning rate decay schedule')
    parser.add_argument('--lr_num_epochs_decay', type=int, default=50, help='LambdaLR: epoch at starting learning rate')
    parser.add_argument('--lr_decay_ratio', type=int, default=500, help='LambdaLR: ratio of linearly decay learning rate to zero')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='adam|rmsprop')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha for rmsprop optimizer')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='weight for adversarial loss')
    parser.add_argument('--lambda_cycle', type=float, default=1.5, help='weight for perceptual loss')
    parser.add_argument('--lambda_idt', type=float, default=10.0, help='weight for identity loss')
    parser.add_argument('--adv_loss_type', type=str, default='rals', help='adversarial Loss: ls|original|hinge|rahinge|rals')
    parser.add_argument('--idt_loss_type', type=str, default='l1', help='identity_loss: l1|l2|smoothl1')
    parser.add_argument('--idt_reset_epoch', type=int, default=20)

    # validation configuration
    parser.add_argument('--num_epochs_start_val', type=int, default=1, help='start validate the model')

    # Directories.
    parser.add_argument('--data_path', type=str, default='dataset dir path')
    parser.add_argument('--save_path', type=str, default='save dir path')

    # Misc
    parser.add_argument("--cuda_num", type=str)
    parser.add_argument('--use_tensorboard', type=str, default=True)
    parser.add_argument('--is_print_network', type=str2bool, default=True)

    return parser.parse_args()