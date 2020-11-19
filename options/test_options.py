"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/',
                            help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=None,
                            help='how many test images to run')
        parser.add_argument('--region_idx', nargs='+', type=int)
        parser.add_argument('--n_interpolation', type=int, default=5)
        parser.add_argument('--n_samples', type=int, default=1)
        parser.add_argument('--noise_delta', type=float, default=0.0)
        parser.add_argument('--noise_dist', type=str, default='normal',
                            help='normal | uniform')
        parser.add_argument('--dont_merge_fake', action='store_true',
                            help="do NOT concat fake along dim 1 for multi_modal etc.")

        parser.add_argument('--manipulate_scale', type=float, default=1.0)

        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(efficient=False)
        parser.set_defaults(max_fm_size=256)
        parser.set_defaults(batchSize=1)
        self.isTrain = False
        return parser
