"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.  Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import os
import pickle
import sys

import torch

import data as data
import deepsee_models
from util import util
from .configurations import get_opt_config

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='8x_independent_128x128',
                            help='name of the experiment. It decides where to store samples and deepsee_models')
        parser.add_argument('--dataset', type=str, default='celebamaskhq',
                            choices=('celeba', 'celebamaskhq'))
        parser.add_argument('--gpu_ids', type=str, default='-1',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='./checkpoints',
                            help='deepsee_models are saved here')
        parser.add_argument('--model', type=str, default='sr',
                            help='which model to use')
        parser.add_argument('--norm_G', type=str,
                            default='spectrallateseansyncbatch3x3',
                            help='instance normalization or batch normalization. spectralinstance | spectralspadesyncbatch3x3 | spectralspadestylesyncbatch3x3 | spectralseansyncbatch3x3 | spectrallateseansyncbatch3x3')
        parser.add_argument('--norm_D', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train',
                            help='train, val, test, etc')
        parser.add_argument('--add_noise', action='store_true',
                            help='add noise to sean')
        parser.add_argument('--noisy_style_scale', type=float, default=0.2,
                            help='Scale of noise for style matrix. Might be zero.')
        parser.add_argument('--noisy_style_dist', type=str, default='uniform',
                            help='distribution of noise applied to the style matrix. Choices: uniform | normal')
        parser.add_argument('--ignore_path_match', action='store_true',
                            help='use amp')

        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=4,
                            help='input batch size')
        parser.add_argument('--preprocess_mode', type=str,
                            default='scale_width_and_crop',
                            help='scaling and cropping of images at load time.',
                            choices=("center_crop_and_resize", "center_crop",
                                     "resize_and_crop", "crop", "scale_width",
                                     "scale_width_and_crop", "scale_shortside",
                                     "scale_shortside_and_crop", "fixed",
                                     "none", "scale_width_and_center_crop"))
        parser.add_argument('--center_crop_size', type=int, default=None,
                            help='178 for CelebA')
        parser.add_argument('--load_size', type=int, default=128,
                            help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=128,
                            help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0,
                            help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=19,
                            help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true',
                            help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels')
        parser.add_argument('--start_size', type=int, default=16,
                            help='For SR model only: Start size (image is downsampled to that size)')
        parser.add_argument('--downscale_label', action="store_true",
                            help='Use downscaled label (same as start_size)')
        parser.add_argument('--max_fm_size', type=int, default=256,
                            help='max feature map size in normalization')
        parser.add_argument('--downsampling_method', type=str,
                            default="bicubic",
                            help='bicubic, linear, nearest')

        # for setting inputs
        parser.add_argument('--dataset_mode', type=str, default='celebamaskhq')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=0, type=int,
                            help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int,
                            default=sys.maxsize,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true',
                            help='load the options from checkpoints and use that as default')
        parser.add_argument('--load_config_from_name', action='store_true',
                            help='load pre-configured options given the name (e.g. 8x_independent_256x256')
        parser.add_argument('--cache_filelist_write', action='store_true',
                            help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true',
                            help='reads from the file list cache')
        parser.add_argument("--identities_file", type=str, default="",
                            required=False)

        # for displays
        parser.add_argument('--display_winsize', type=int, default=400,
                            help='display window size')

        # for deepsee_models
        parser.add_argument('--netG', type=str, default='deepsee',
                            help='selects model to use for super-resolution')
        parser.add_argument('--netE', type=str, default='combinedstyle',
                            help='selects model to use for netE. none | combinedstyle | fullstyle')
        parser.add_argument('--ngf', type=int, default=32,
                            help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')
        parser.add_argument('--regional_style_size', type=int, default=128,
                            help="dimension of the latent z vector")
        parser.add_argument('--full_style_image', action='store_true',
                            help="Whether to feed the full style image")
        parser.add_argument('--guiding_style_image', action='store_true',
                            help="Use a full image, but another image from the same person")
        parser.add_argument('--guiding_style_image2', action='store_true',
                            help="Use a full image, but another image from the same person")
        parser.add_argument('--random_style_matrix', action='store_true',
                            help="Create a random style matrix from N(0,1)")
        parser.add_argument('--gpu_info', action='store_true',
                            help="print gpu info")
        parser.add_argument('--model_parallel_mode', type=int, default=0,
                            help="1 for 512x512 deepsee_models")
        parser.add_argument('--ablation', type=str, default="",
                            help="nospadenostyle")

        # for instance-wise features
        parser.add_argument('--nef', type=int, default=32,
                            help='# of encoder filters in the first conv layer')
        # Google Sheets related
        parser.add_argument('--gsheet_secrets_json_file', type=str, help='')
        parser.add_argument('--gsheet_workbook_key', type=str,
                            default='1byolOn6WRst4lBiQzHQV66f4oluEESur6D7q8nPLEio')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = deepsee_models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write(
                    '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
                          (1 if opt.contain_dontcare_label else 0)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt

        if self.opt.load_config_from_name:
            # We add the pre-configured options based on the name
            self.opt = get_opt_config(self.opt, self.opt.name)
        return self.opt
