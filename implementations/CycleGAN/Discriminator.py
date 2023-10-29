import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

##############################Discriminator################################
class MultiScaleDiscriminator_Scale4_1(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(MultiScaleDiscriminator_Scale4_1, self).__init__()

        # scale 1
        #d_1: [ B, 1, 257, 189] → [100, conv_dim, 128, 94]              d_1_pred: [100, conv_dim, 128, 94] → [100, 1, 128, 94]
        d_1 = [dis_conv_block(in_channels=1, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 2
        #d_2: [100, 64, conv_dim, 94] → [100, conv_dim*2, 64, 47]              d_2_pred: [100, conv_dim*2, 64, 47] → [100, 1, 64, 47]
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=7, stride=2, padding=3, dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]
        d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 3
        #d_3: [100, conv_dim*2, 64, 47] → [100, conv_dim*4, 32, 23]              d_3_pred: [100, conv_dim*4, 32, 23] → [100, 1, 32, 23]
        d_3 = [dis_conv_block(in_channels=conv_dim* 2, out_channels=conv_dim* 4, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim* 4, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 4
        #d_4: [100, conv_dim*4, 32, 23] → [100, conv_dim*8, 16, 11]              d_4_pred: [100, conv_dim*8, 16, 11] → [100, 1, 16, 11]
        d_4 = [dis_conv_block(in_channels=conv_dim* 4, out_channels=conv_dim* 8, kernel_size=5, stride=32, padding=2, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=1, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)

    def forward(self, x):
        x = torch.abs(x).to(dtype=torch.float, non_blocking=True)
        x = torch.unsqueeze(x,1)
        ds1 = self.d1(x)
        ds1_pred = self.d1_pred(ds1)

        ds2 = self.d2(ds1)
        ds2_pred = self.d2_pred(ds2)

        ds3 = self.d3(ds2)
        ds3_pred = self.d3_pred(ds3)

        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)

        return [ds1_pred, ds2_pred, ds3_pred, ds4_pred]

class MultiScaleDiscriminator_Scale4_4(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(MultiScaleDiscriminator_Scale4_4, self).__init__()

        # scale 1
        #d_1: [ B, 1, 257, 189] → [100, conv_dim, 128, 94]              d_1_pred: [100, conv_dim, 128, 94] → [100, 1, 128, 94]
        d_1 = [dis_conv_block(in_channels=1, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 2
        #d_2: [100, 64, conv_dim, 94] → [100, conv_dim*2, 64, 47]              d_2_pred: [100, conv_dim*2, 64, 47] → [100, 1, 64, 47]
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=7, stride=2, padding=3, dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]
        d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 3
        #d_3: [100, conv_dim*2, 64, 47] → [100, conv_dim*4, 32, 23]              d_3_pred: [100, conv_dim*4, 32, 23] → [100, 1, 32, 23]
        d_3 = [dis_conv_block(in_channels=conv_dim* 2, out_channels=conv_dim* 4, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim* 4, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 4
        #d_4: [100, conv_dim*4, 32, 23] → [100, conv_dim*8, 16, 11]              d_4_pred: [100, conv_dim*8, 16, 11] → [100, 1, 16, 11]
        d_4 = [dis_conv_block(in_channels=conv_dim* 4, out_channels=conv_dim* 8, kernel_size=5, stride=8, padding=2, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)

    def forward(self, x):
        x = torch.abs(x).to(dtype=torch.float, non_blocking=True)
        x = torch.unsqueeze(x,1)
        ds1 = self.d1(x)
        ds1_pred = self.d1_pred(ds1)

        ds2 = self.d2(ds1)
        ds2_pred = self.d2_pred(ds2)

        ds3 = self.d3(ds2)
        ds3_pred = self.d3_pred(ds3)

        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)

        return [ds1_pred, ds2_pred, ds3_pred, ds4_pred]

class MultiScaleDiscriminator_Scale4_8(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(MultiScaleDiscriminator_Scale4_8, self).__init__()

        # scale 1
        #d_1: [ B, 1, 257, 189] → [100, conv_dim, 128, 94]              d_1_pred: [100, conv_dim, 128, 94] → [100, 1, 128, 94]
        d_1 = [dis_conv_block(in_channels=1, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 2
        #d_2: [100, 64, conv_dim, 94] → [100, conv_dim*2, 64, 47]              d_2_pred: [100, conv_dim*2, 64, 47] → [100, 1, 64, 47]
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=7, stride=2, padding=3, dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]
        d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 3
        #d_3: [100, conv_dim*2, 64, 47] → [100, conv_dim*4, 32, 23]              d_3_pred: [100, conv_dim*4, 32, 23] → [100, 1, 32, 23]
        d_3 = [dis_conv_block(in_channels=conv_dim* 2, out_channels=conv_dim* 4, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim* 4, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 4
        #d_4: [100, conv_dim*4, 32, 23] → [100, conv_dim*8, 16, 11]              d_4_pred: [100, conv_dim*8, 16, 11] → [100, 1, 16, 11]
        d_4 = [dis_conv_block(in_channels=conv_dim* 4, out_channels=conv_dim* 8, kernel_size=5, stride=4, padding=2, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)

    def forward(self, x):
        x = torch.abs(x).to(dtype=torch.float, non_blocking=True)
        x = torch.unsqueeze(x,1)
        ds1 = self.d1(x)
        ds1_pred = self.d1_pred(ds1)

        ds2 = self.d2(ds1)
        ds2_pred = self.d2_pred(ds2)

        ds3 = self.d3(ds2)
        ds3_pred = self.d3_pred(ds3)

        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)

        return [ds1_pred, ds2_pred, ds3_pred, ds4_pred]

class MultiScaleDiscriminator_Scale4_31_22(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(MultiScaleDiscriminator_Scale4_31_22, self).__init__()

        # scale 1
        #d_1: [100, 1, 257, 189] → [100, 64, 128, 94]              d_1_pred: [100, 64, 128, 94] → [100, 1, 128, 94]
        d_1 = [dis_conv_block(in_channels=1, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 2
        #d_2: [100, 64, 128, 94] → [100, 128, 64, 47]              d_2_pred: [100, 128, 64, 47] → [100, 1, 64, 47]
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 3
        #d_3: [100, 128, 64, 47] → [100, 256, 32, 23]              d_3_pred: [100, 256, 32, 23] → [100, 1, 32, 23]
        d_3 = [dis_conv_block(in_channels=conv_dim* 2, out_channels=conv_dim* 4, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim* 4, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 4
        #d_4: [100, 256, 32, 23] → [100, 512, 31, 22]              d_4_pred: [100, 512, 16, 11] → [100, 1, 31, 22]
        d_4 = [dis_conv_block(in_channels=conv_dim* 4, out_channels=conv_dim* 8, kernel_size=4, stride=1, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)

    def forward(self, x):
        x = torch.abs(x).to(dtype=torch.float, non_blocking=True)
        x = torch.unsqueeze(x,1)
        ds1 = self.d1(x)
        ds1_pred = self.d1_pred(ds1)

        ds2 = self.d2(ds1)
        ds2_pred = self.d2_pred(ds2)

        ds3 = self.d3(ds2)
        ds3_pred = self.d3_pred(ds3)

        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)

        return [ds1_pred, ds2_pred, ds3_pred, ds4_pred]

class MultiScaleDiscriminator_Scale4_62_45(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(MultiScaleDiscriminator_Scale4_62_45, self).__init__()

        # scale 1
        #d_1: [ B, 1, 257, 189] → [100, conv_dim, 128, 94]              d_1_pred: [100, conv_dim, 128, 94] → [100, 1, 128, 94]
        d_1 = [dis_conv_block(in_channels=1, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 2
        #d_2: [100, 64, conv_dim, 94] → [100, conv_dim*2, 64, 47]              d_2_pred: [100, conv_dim*2, 64, 47] → [100, 1, 64, 47]
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=7, stride=2, padding=3, dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]
        d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 3
        ##d_3: [100, conv_dim*2, 64, 47] → [100, conv_dim*4, 32, 23]              d_3_pred: [100, conv_dim*4, 32, 23] → [100, 1, 32, 23]
        #d_3: [100, 128, 64, 47] → [100, 256, 63, 46]              d_3_pred: [100, 256, 63, 46] → [100, 1, 63, 46]
        d_3 = [dis_conv_block(in_channels=conv_dim* 2, out_channels=conv_dim* 4, kernel_size=4, stride=1, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim* 4, out_channels=1, kernel_size=7, stride=1, padding=1, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 4
        #d_4: [100, 256, 63, 46] → [100, 512, 62, 45]              d_4_pred: [100, 512, 62, 45] → [100, 1, 62, 45]
        d_4 = [dis_conv_block(in_channels=conv_dim* 4, out_channels=conv_dim* 8, kernel_size=4, stride=1, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)

    def forward(self, x):
        x = torch.abs(x).to(dtype=torch.float, non_blocking=True)
        x = torch.unsqueeze(x,1)
        ds1 = self.d1(x)
        ds1_pred = self.d1_pred(ds1)

        ds2 = self.d2(ds1)
        ds2_pred = self.d2_pred(ds2)

        ds3 = self.d3(ds2)
        ds3_pred = self.d3_pred(ds3)

        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)

        return [ds1_pred, ds2_pred, ds3_pred, ds4_pred]

def dis_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun, use_sn):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn))
    norm_fun = get_norm_fun(norm_fun)
    main.append(norm_fun(out_channels))
    main.append(get_act_fun(act_fun))
    main = nn.Sequential(*main)
    return main

def dis_pred_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, type):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias))
    if type in ['ls', 'rals']:
        main.append(nn.Sigmoid())
    elif type in ['hinge', 'rahinge']:
        main.append(nn.Tanh())
    else:
        raise NotImplementedError("Adversarial loss [{}] is not found".format(type))
    main = nn.Sequential(*main)
    return main

def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_act_fun(act_fun_type='LeakyReLU'):
    if isinstance(act_fun_type, str):
        if act_fun_type == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun_type == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun_type == 'Swish':
            return Swish()
        elif act_fun_type == 'SELU':
            return nn.SELU(inplace=True)
        elif act_fun_type == 'none':
            return nn.Sequential()
        else:
            raise NotImplementedError('activation function [%s] is not found' % act_fun_type)
    else:
        return act_fun_type()