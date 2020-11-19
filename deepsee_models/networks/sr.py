
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from deepsee_models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from deepsee_models.networks.base_network import BaseNetwork
from util.util import gpu_info


class DeepSEESR(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        nf = opt.ngf
        self.start_size = opt.start_size
        self.n_blocks = int(np.log2(opt.crop_size) - np.log2(self.start_size))

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.initial = nn.Conv2d(3, 16 * nf, 3, padding=1)

        early_style = not ("late" in self.opt.norm_G)
        # We do not use style in the early layers
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, style=early_style)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, style=True)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt, style=True)

        self.mp = self.opt.model_parallel_mode
        # TODO: remove this ugly hack (cropped in opt.name)
        max_n_blocks = 99
        if self.opt.load_size >= 512:
            max_n_blocks = 4
        # max_n_blocks = 4 if (self.mp > 0 or "cropped" in self.opt.name) else 99  # We only reduce the number of full resnet blocks if we use
        # model parallelism
        upscaling_modules = [SPADEResnetBlock(16 * nf, 16 * nf, opt, style=True) for i in range(1, min(self.n_blocks, max_n_blocks))]
        if max_n_blocks != 99:
            for i in range(max_n_blocks, self.n_blocks): # 512 and 1024
                # We use puresean blocks here
                upscaling_modules.append(SPADEResnetBlock(16 * nf, 16 * nf, opt, style=True, puresean=True))
        self.up_list = nn.ModuleList(upscaling_modules)

        final_nc = 16*nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def get_device(self):
        return self.initial.weight.device

    def forward(self, x_downsized, seg=None, z=None):
        split_location = -1
        gpu_info("SR start", self.opt)
        x = self.initial(x_downsized)

        x = self.head_0(x, seg, z)

        x = self.up(x)
        x = self.G_middle_0(x, seg, z)
        x = self.G_middle_1(x, seg, z)
        for i in range(self.n_blocks - 1):
            if self.mp == 1 and i == 3:
                split_location = 1
            if self.mp == 2 and i == 3:
                self.up_list[i] = self.up_list[i].cuda(1)
                x = x.cuda(1)
                seg = seg.cuda(1)
                z = z.cuda(1)
            if self.mp >= 2 and i == 4:
                self.up_list[i] = self.up_list[i].cuda(2)
                x = x.cuda(2)
                seg = seg.cuda(2)
                z = z.cuda(2)
                split_location = 2

            x = self.up(x)
            x = self.up_list[i](x, seg, z, split_location=split_location)
            gpu_info("SR after up {}".format(i), self.opt)

        if self.mp > 0:
            x = x.cuda(0)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        gpu_info("SR end", self.opt)
        return x

