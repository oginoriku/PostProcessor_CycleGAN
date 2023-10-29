import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

#U-net Generator
class U_Net(torch.nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        # U-NetのEocoder部分
        self.down0 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.down1 = self.__encoder_block(64, 128)
        self.down2 = self.__encoder_block(128, 256)
        self.down3 = self.__encoder_block(256, 512)
        self.down4 = self.__encoder_block(512, 512)
        self.down5 = self.__encoder_block(512, 512)
        self.down6 = self.__encoder_block(512, 512)
        self.down7 = self.__encoder_block(512, 512, use_norm=False, kernel_size=(4, 3), stride=2, padding=1)
        # U-NetのDecoder部分
        self.up7 = self.__decoder_block(512, 512, kernel_size=(4,3), stride=2, padding=1)
        self.up6 = self.__decoder_block(1024, 512, use_dropout=True)
        self.up5 = self.__decoder_block(1024, 512, use_dropout=True, kernel_size=(4,5), stride=2, padding=1)
        self.up4 = self.__decoder_block(1024, 512, use_dropout=True, kernel_size=(4,5), stride=2, padding=1)
        self.up3 = self.__decoder_block(1024, 256, kernel_size=(4,5), stride=2, padding=1)
        self.up2 = self.__decoder_block(512, 128, kernel_size=(4,5), stride=2, padding=1)
        self.up1 = self.__decoder_block(256, 64)
        # Gの最終出力
        self.up0 = nn.Sequential(
            self.__decoder_block(128, 1, use_norm=False, kernel_size=(5,5), stride=2, padding=1),
            nn.ReLU(),
        )

    def __encoder_block(self, input, output, use_norm=True, kernel_size=4, stride=2, padding=1):
        # LeakyReLU＋Downsampling
        layer = [
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input, output, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        # BatchNormalization
        if use_norm:
            layer.append(nn.BatchNorm2d(output))
        return nn.Sequential(*layer)

    def __decoder_block(self, input, output, use_norm=True, use_dropout=False, kernel_size=4, stride=2, padding=1):
        # ReLU＋Upsampling
        layer = [
            nn.ReLU(True),
            nn.ConvTranspose2d(input, output, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        # BachNormalization
        if use_norm:
            layer.append(nn.BatchNorm2d(output))
        # Dropout
        if use_dropout:
            layer.append(nn.Dropout(0.5))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = torch.abs(x).to(dtype=torch.float, non_blocking=True)
        x = torch.unsqueeze(x,1)
        # 偽物画像の生成
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        y7 = self.up7(x7)
        # Encoderの出力をDecoderの入力にSkipConnectionで接続
        y6 = self.up6(self.concat(x6, y7))
        y5 = self.up5(self.concat(x5, y6))
        y4 = self.up4(self.concat(x4, y5))
        y3 = self.up3(self.concat(x3, y4))
        y2 = self.up2(self.concat(x2, y3))
        y1 = self.up1(self.concat(x1, y2))
        y0 = self.up0(self.concat(x0, y1))
        y = torch.squeeze(y0,1)

        return y

    def concat(self, x, y):
        # 特徴マップの結合
        return torch.cat([x, y], dim=1)
