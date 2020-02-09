"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# from basenet.vgg16_bn import vgg16_bn, init_weights
from .model import EfficientNet

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.eff = EfficientNet.from_name('efficientnet-b0')#.cuda()
        # self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1280, 320, 320)
        self.upconv2 = double_conv(320, 112, 112)
        self.upconv3 = double_conv(112, 40, 40)
        self.upconv4 = double_conv(40, 24, 24)
        self.upconv5 = double_conv(24, 16, 16)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.upconv5.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ Base network """
        # sources = self.basenet(x)
        _,sources = self.eff.extract_features(x)

        sources.reverse()
        # print(1,sources[0].shape)
        # print(2,sources[1].shape)
        # print(3,sources[2].shape)
        # print(4,sources[3].shape)
        # print(5.1,sources[4].shape)
        # print(5.2,sources[5].shape)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        # print(6,y.shape)
        y = self.upconv1(y)
        # print(7,y.shape)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        # print(8,y.shape)
        y = torch.cat([y, sources[2]], dim=1)
        # print(9,y.shape)
        y = self.upconv2(y)
        # print(10,y.shape)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        # print(11,y.shape)
        y = torch.cat([y, sources[3]], dim=1)
        # print(12,y.shape)
        y = self.upconv3(y)
        # print(13,y.shape)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        # print(14,y.shape)
        y = torch.cat([y, sources[4]], dim=1)
        # print(15,y.shape)
        y = self.upconv4(y)
        # print(16,y.shape)

        y = F.interpolate(y, size=sources[5].size()[2:], mode='bilinear', align_corners=False)
        # print(17,y.shape)
        y = torch.cat([y, sources[5]], dim=1)
        # print(18,y.shape)
        feature = self.upconv5(y)
        # print(19,y.shape)

        y = self.conv_cls(feature)
        # print(20,y.shape)

        return y.permute(0,2,3,1), feature


if __name__ == '__main__':
    # model = CRAFT(pretrained=True)#.cuda()
    # output, _ = model(torch.randn(1, 3, 768, 768))#.cuda())
    # print(output.shape)

    model2 =CRAFT(pretrained=False)
    output, _ = model2(torch.randn(1, 3, 768, 768))#.cuda())
    print(output.shape)