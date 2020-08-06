"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.  Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from deepsee_models.networks.normalization import SPADE, SEAN_Block, NoiseInjection, PureSEAN_Block
from torch.utils.checkpoint import checkpoint

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
from util.util import gpu_info


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, style=True, puresean=False):
        super().__init__()
        self.opt = opt
        self.efficient = self.opt.efficient
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        NormBlock = self._get_block(spade_config_str, style, puresean)
        self.norm_0 = NormBlock(spade_config_str, fin, opt.semantic_nc, opt)
        self.norm_1 = NormBlock(spade_config_str, fmiddle, opt.semantic_nc, opt)
        if self.learned_shortcut:
            self.norm_s = NormBlock(spade_config_str, fin, opt.semantic_nc, opt)

        # Do not add noise when we are in eval() mode (self.training=False)
        if self.add_noise:
            self.noise_in = NoiseInjection(fin)
            self.noise_skip = NoiseInjection(fin)
            self.noise_middle = NoiseInjection(fmiddle)

    @property
    def add_noise(self):
        # self.training might be false only after instantiation, so we need to check this here.
        return self.opt.add_noise and self.training

    def _get_block(self, config_text, style, puresean=False):
        if puresean:
            return PureSEAN_Block
        elif style:
            if "sean" in config_text:
                return SEAN_Block
        return SPADE

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, style=None, split_location=-1):
        if self.add_noise:
            x = self.noise_in(x)
        # style can be None
        x_s = self.shortcut(x, seg, style)
        #
        # # gpu_info("resblk line 84", self.opt)
        if split_location == 1:
            x = x.cuda(1)
            seg = seg.cuda(1)
            style = style.cuda(1)
            self.norm_0 = self.norm_0.cuda(1)
            self.conv_0 = self.conv_0.cuda(1)
            self.norm_1 = self.norm_1.cuda(1)

        gpu_info("resblk line 94", self.opt)

        x = self.actvn(self.norm_0(x, seg, style))
        gpu_info("resblk line 96", self.opt)

        if self.efficient:
            dx = checkpoint(self.conv_0, x)
        else:
            dx = self.conv_0(x)

        gpu_info("resblk line 104", self.opt)
        if split_location == 2:
            dx = dx.cuda(3)
            seg = seg.cuda(3)
            style = style.cuda(3)
            self.norm_1 = self.norm_1.cuda(3)
            self.conv_1 = self.conv_1.cuda(3)
            if self.add_noise:
                self.noise_middle = self.noise_middle.cuda(3)

        gpu_info("resblk line 99", self.opt)
        if self.add_noise:
            dx = self.noise_middle(dx)

        dx = self.actvn(self.norm_1(dx, seg, style))

        if split_location == 1:
            dx = dx.cuda(0)

        if self.efficient:
            dx = checkpoint(self.conv_1, dx)
        else:
            dx = self.conv_1(dx)

        if split_location == 2:
            x_s = x_s.cuda(3)

        out = x_s + dx
        # if split_location == 1:
        #     out = out.cuda(0)
        return out

    def shortcut(self, x, seg, style=None):
        if self.add_noise:
            x = self.noise_skip(x)

        if self.learned_shortcut:
            x = self.norm_s(x, seg, style)
            if self.efficient:
                x_s = checkpoint(self.conv_s, x)
            else:
                x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out




