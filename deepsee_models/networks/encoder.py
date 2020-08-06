from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from deepsee_models.networks.base_network import BaseNetwork
from deepsee_models.networks.normalization import get_nonspade_norm_layer
from util.util import gpu_info


class AbtractStyleEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.kw = 3
        self.pw = int(np.ceil((self.kw - 1.0) / 2))
        self.nf = opt.nef
        self.out_size = opt.regional_style_size
        self.norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)

        # TODO: move this to subclasses (otherwise we have it twice)
        self.final = nn.Sequential(
            self.norm_layer(nn.Conv2d(self.nf * 8, self.out_size, self.kw, stride=1, padding=self.pw)),
            nn.Tanh()
        )

    def forward_main(self, x=None):
        activations = OrderedDict()
        for name, module in self.layers.items():
            x = module(x)
            activations[name] = x
        return x, activations

    def extract_style_matrix(self, x, seg):
        if seg.size(2) != x.size(2) or seg.size(3) != x.size(3):
            seg = F.interpolate(seg, size=(x.size(2), x.size(3)), mode='nearest')

        # seg.size(1) is the number of semantic regions (19 for CelebAMask-HQ)
        x_expanded = x.unsqueeze(1).expand(-1, seg.size(1), -1, -1, -1)  # Shape BS, label_nc, C, H, W
        # x_expanded.size(-3) is the number of channels
        seg_expanded = seg.unsqueeze(2).expand(-1, -1, x_expanded.shape[-3], -1, -1) # Shape BS, label_nc, 512, H, W
        # seg_expanded contains binary maps. Hence, we can extract semantic regions by multiplying
        # Now we have the activations for each semantic regions
        combined = x_expanded * seg_expanded
        # We collapse height and width by averaging
        style_matrix = combined.mean(-1).mean(-1)  # Shape BS, label_nc, 512
        return style_matrix

    def corrupt_style_matrix(self, style_matrix, max_range_noise, region_idx=None):
        if not region_idx:
            # All regions are indexed
            region_idx = list(range(style_matrix.size(1)))

        noise_weights = self.actv_weights(self.noise_weights[region_idx]).unsqueeze(0).unsqueeze(-1).expand_as(
            style_matrix[:, region_idx])
        # rand samples from [0,1], so we adjust the range to [-1, 1]
        if self.opt.noisy_style_dist == 'uniform':
            noise = ((torch.rand_like(style_matrix[:, region_idx],
                                      device=style_matrix.device) * 2) - 1) * max_range_noise
        elif self.opt.noisy_style_dist == 'normal':
            noise = ((torch.randn_like(style_matrix[:, region_idx],
                                       device=style_matrix.device) * 2) - 1) * max_range_noise
        else:
            raise ValueError("Does not exist: {}".format(self.opt.noisy_style_dist))

        style_matrix[:, region_idx] = (style_matrix[:, region_idx] + noise * noise_weights).clamp(-1,
                                                                                                  1)  # We want to make sure to be in this range
        return style_matrix


class FullStyleEncoder(AbtractStyleEncoder):
    """ Following SEAN paper  https://arxiv.org/pdf/1911.12861.pdf"""

    def __init__(self, opt):
        super().__init__(opt)
        input_nc = opt.label_nc if opt.random_style_matrix else 3

        # TO check:
        # Norm layers?
        self.layers = OrderedDict()
        self.layers["initial"] = nn.Sequential(
                self.norm_layer(nn.Conv2d(input_nc, self.nf, self.kw, stride=1, padding=self.pw)),
                nn.LeakyReLU(0.2, False)
        )
        self.layers["down0"] = nn.Sequential(
            self.norm_layer(nn.Conv2d(self.nf * 1, self.nf * 2, self.kw, stride=2, padding=self.pw)),
            nn.LeakyReLU(0.2, False)
        )
        self.layers["down1"] = nn.Sequential(
            self.norm_layer(nn.Conv2d(self.nf * 2, self.nf * 4, self.kw, stride=2, padding=self.pw)),
            nn.LeakyReLU(0.2, False)
        )
        self.layers["up_conv"] = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.norm_layer(nn.Conv2d(self.nf * 4, self.nf * 8, self.kw, padding=self.pw)),
            nn.LeakyReLU(0.2, False)
        )

        for name, module in self.layers.items():
            self.add_module(name, module)
        # Difference to SEAN: we do not use TConv, but upsample and run conv as suggested here
        # https://distill.pub/2016/deconv-checkerboard/
        # Odena, et al., "Deconvolution and Checkerboard Artifacts", Distill, 2016. http://doi.org/10.23915/distill.00003
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # We do not want to add extra noise here in the case of the combined style encoer
        self.noisy_style = "fullstyle" in self.opt.netE and self.opt.noisy_style_scale > 0
        if self.noisy_style:
            # We have one weight for each region
            self.noise_weights = nn.Parameter(torch.zeros(opt.label_nc), requires_grad=True)
            self.actv_weights = nn.Sigmoid()
            self.max_range_noise = self.opt.noisy_style_scale  # The noise can change the style matrix max +- this value

    def forward(self, x=None, seg=None, mode="full", no_noise=False):
        gpu_info("full style encoder start", self.opt)
        if self.opt.random_style_matrix:
            x = torch.randn((seg.size(0), seg.size(1), self.opt.crop_size, self.opt.crop_size), device=seg.device)
            x = x * seg  # We set the unused regions to zero.

        x, activations = self.forward_main(x)
        x = self.final(x)
        # x has new height and width so we need to adjust the segmask size
        if seg.size(2) != x.size(2) or seg.size(3) != x.size(3):
            seg = F.interpolate(seg, size=(x.size(2), x.size(3)), mode='nearest')

        style_matrix = self.extract_style_matrix(x, seg)
        if self.noisy_style and not no_noise:  # For reconstruction or also for demo we want no noise
            style_matrix = self.corrupt_style_matrix(style_matrix, self.max_range_noise)
        gpu_info("full style encoder end", self.opt)
        return style_matrix, activations


class MinistyleEncoder(AbtractStyleEncoder):
    def __init__(self, opt):
        super().__init__(opt)

        # TO check:
        # Norm layers?
        self.layers = OrderedDict()
        self.layers["initial"] = nn.Sequential(
                self.norm_layer(nn.Conv2d(3, self.nf, self.kw, stride=1, padding=self.pw)),
                nn.LeakyReLU(0.2, False)
        )
        self.layers["conv0"] = nn.Sequential(
            self.norm_layer(nn.Conv2d(self.nf * 1, self.nf * 2, self.kw, stride=1, padding=self.pw)),
            nn.LeakyReLU(0.2, False)
        )
        self.layers["conv1"] = nn.Sequential(
            self.norm_layer(nn.Conv2d(self.nf * 2, self.nf * 4, self.kw, stride=1, padding=self.pw)),
            nn.LeakyReLU(0.2, False)
        )
        self.layers["conv2"] = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.norm_layer(nn.Conv2d(self.nf * 4, self.nf * 8, self.kw, padding=self.pw)),
            nn.LeakyReLU(0.2, False)
        )

        for name, module in self.layers.items():
            self.add_module(name, module)
        self.opt = opt

    def forward(self, x=None, seg=None, mode="mini"):
        gpu_info("Mini style encoder start", self.opt)
        x, activations = self.forward_main(x)

        x = self.final(x)
        # x has new height and width so we need to adjust the segmask size
        if seg.size(2) != x.size(2) or seg.size(3) != x.size(3):
            seg = F.interpolate(seg, size=(x.size(2), x.size(3)), mode='nearest')

        style_matrix = self.extract_style_matrix(x, seg)
        gpu_info("Mini style encoder end", self.opt)
        return style_matrix, activations


class CombinedstyleEncoder(AbtractStyleEncoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.encoder_full = FullStyleEncoder(opt)
        self.encoder_mini = MinistyleEncoder(opt)

        self.final = nn.Sequential(
            self.norm_layer(nn.Conv2d(self.nf * 8, self.out_size, self.kw, stride=1, padding=self.pw)),
            nn.Tanh()
        )
        self.noisy_style = self.opt.noisy_style_scale > 0
        if self.noisy_style:
            # We have one weight for each region
            self.noise_weights = nn.Parameter(torch.zeros(opt.label_nc), requires_grad=True)
            self.actv_weights = nn.Sigmoid()
            self.max_range_noise = self.opt.noisy_style_scale  # The noise can change the style matrix max +- this value

    def forward(self, x=None, seg=None, mode=None, no_noise=False):
        gpu_info("Combined encoder start", self.opt)
        if mode == "full":
            x, activations = self.encoder_full.forward_main(x)
        elif mode == "mini":
            x, activations = self.encoder_mini.forward_main(x)
        else:
            raise NotImplementedError()
        x = self.final(x)
        # Shared layer
        style_matrix = self.extract_style_matrix(x, seg)
        if self.noisy_style and not no_noise:
            style_matrix = self.corrupt_style_matrix(style_matrix, self.max_range_noise)

        gpu_info("Combined encoder end", self.opt)
        return style_matrix, activations
