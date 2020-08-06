"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.  Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsee_models.networks.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d
import torch.nn.utils.spectral_norm as spectral_norm
from torch.utils.checkpoint import checkpoint

# Returns a function that creates a normalization function
# that does not condition on semantic map
from util.util import gpu_info


def get_nonspade_norm_layer(opt, norm_type='instance', oneD=False):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        out_channel = get_out_channel(layer)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm1d(out_channel, affine=True) if oneD else nn.BatchNorm2d(out_channel, affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm1d(out_channel, affine=True) if oneD else  SynchronizedBatchNorm2d(out_channel, affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm1d(out_channel, affine=False) if oneD else nn.InstanceNorm2d(out_channel, affine=False)  # Why is this not affine?
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
torch.autograd.set_detect_anomaly(True)
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, opt):
        super().__init__()
        # spadestylesyncbatch3x3
        assert config_text.startswith('spade') or config_text.startswith('latesean')
        if config_text.startswith('latesean'):
            parsed = re.search('latesean(\D+)(\d)x\d', config_text)
        else:
            parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.nc = label_nc

        if 'instance' in param_free_norm_type:
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif 'syncbatch' in param_free_norm_type:
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif 'batch' in param_free_norm_type:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, style=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        # Original way of SPADE
        actv = segmap
        actv = self.mlp_shared(actv)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class SEAN_Block(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, opt):
        super().__init__()
        self.opt = opt
        self.efficient = opt.efficient
        # spadestylesyncbatch3x3
        assert 'sean' in config_text
        parsed = re.search('sean(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.nc = label_nc
        self.style_size = opt.regional_style_size
        if 'instance' in param_free_norm_type:
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif 'syncbatch' in param_free_norm_type:
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif 'batch' in param_free_norm_type:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        self.style_conv = nn.Conv1d(19, 19, kernel_size=1)

        self.mlp_style_gamma = nn.Conv2d(self.style_size, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_style_beta = nn.Conv2d(self.style_size, norm_nc, kernel_size=ks, padding=pw)

        self.alpha_beta = nn.Parameter(torch.rand(1), requires_grad=True)
        self.alpha_gamma = nn.Parameter(torch.rand(1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

        self.mp = self.opt.model_parallel_mode

    def forward(self, x, segmap, style=None):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        out_size = x.size()[2:]
        fm_size = [min(s, self.opt.max_fm_size) for s in out_size]

        segmap = F.interpolate(segmap, size=fm_size, mode='nearest')
        actv = self.mlp_shared(segmap)

        # Part 2. produce scaling and bias conditioned on semantic map and style (SEAN style part)
        H, W = segmap.shape[-2:]
        # style is BSx19x512

        # Maybe this block could also be working on feature maps of smaller size for less memory usage.
        style_expanded = style.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
        segmap_expanded = segmap.unsqueeze(2).expand(-1, -1, self.style_size, -1, -1)
        # segmap is binary, so we can use it to style particular regions
        style_map = (style_expanded * segmap_expanded).sum(1)
        # This should not change any values, but collapse the channel
        # Specifically compare values because out_size is a tensor
        if out_size[0] != fm_size[0] or out_size[1] != fm_size[1]:
            actv = F.interpolate(actv, size=out_size)
            style_map = F.interpolate(actv, size=out_size)

        if self.efficient:
            gamma = checkpoint(self.mlp_gamma,actv)
            beta = checkpoint(self.mlp_beta,actv)
            beta_style = checkpoint(self.mlp_style_beta, style_map)
            gamma_style = checkpoint(self.mlp_style_gamma, style_map)
        else:
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)
            beta_style = self.mlp_style_beta(style_map)
            gamma_style = self.mlp_style_gamma(style_map)

        # Part 3. produce scaling and bias conditioned on semantic map (SPADE)
        # SPADE

        # Part 4. Combine

        weight_alpha_beta = self.sigmoid(self.alpha_beta)
        weight_alpha_gamma = self.sigmoid(self.alpha_gamma)
        # gpu_info("269", self.opt)
        combined_offset = (weight_alpha_beta * beta_style + (1.0 - weight_alpha_beta) * beta)
        combined_scale = (weight_alpha_gamma * gamma_style + (1.0 - weight_alpha_gamma) * gamma + 1)
        return normalized * combined_scale + combined_offset


class PureSEAN_Block(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, opt):
        super().__init__()
        self.opt = opt
        self.efficient = opt.efficient
        # spadestylesyncbatch3x3
        assert 'sean' in config_text
        parsed = re.search('sean(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.nc = label_nc
        self.style_size = opt.regional_style_size
        if 'instance' in param_free_norm_type:
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif 'syncbatch' in param_free_norm_type:
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif 'batch' in param_free_norm_type:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.style_conv = nn.Conv1d(19, 19, kernel_size=1)

        self.mlp_style_gamma = nn.Conv2d(self.style_size, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_style_beta = nn.Conv2d(self.style_size, norm_nc, kernel_size=ks, padding=pw)

        self.mp = self.opt.model_parallel_mode

    def forward(self, x, segmap, style=None):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        out_size = x.size()[2:]
        fm_size = [min(s, self.opt.max_fm_size) for s in out_size]

        segmap = F.interpolate(segmap, size=fm_size, mode='nearest')
        actv = self.mlp_shared(segmap)

        # Part 2. produce scaling and bias conditioned on semantic map and style (SEAN style part)
        H, W = segmap.shape[-2:]
        # style is BSx19x512

        # Maybe this block could also be working on feature maps of smaller size for less memory usage.
        style_expanded = style.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
        segmap_expanded = segmap.unsqueeze(2).expand(-1, -1, self.style_size, -1, -1)
        # segmap is binary, so we can use it to style particular regions
        style_map = (style_expanded * segmap_expanded).sum(1)
        # This should not change any values, but collapse the channel
        # Specifically compare values because out_size is a tensor
        if out_size[0] != fm_size[0] or out_size[1] != fm_size[1]:
            actv = F.interpolate(actv, size=out_size)
            style_map = F.interpolate(actv, size=out_size)

        if self.efficient:
            beta_style = checkpoint(self.mlp_style_beta, style_map)
            gamma_style = checkpoint(self.mlp_style_gamma, style_map)
        else:
            beta_style = self.mlp_style_beta(style_map)
            gamma_style = self.mlp_style_gamma(style_map)

        return normalized * gamma_style + beta_style


class NoiseInjection(nn.Module):
    """
    Source:
    https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
    """
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.weight = nn.Parameter(torch.zeros(self.n_channels), requires_grad=True)

    def forward(self, tensor, noise=None):
        if noise is None:
            batch, _, height, width = tensor.shape
            noise = tensor.new_empty(batch, self.n_channels, height, width).normal_()
        return tensor + \
               self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, tensor.size(-2), tensor.size(-1)) * noise
