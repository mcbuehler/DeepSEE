"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.  Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=20000,
                            help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1000,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=3000,
                            help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true',
                            help='only do one epoch and displays at each iteration')

        # for training
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=50,
                            help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=25,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0,
                            help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9,
                            help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true',
                            help='Use TTUR training scheme')
        parser.add_argument('--efficient', action='store_true',
                            help='Use gradient checkpointing for memory-efficient training (at the cost of time). Only use this option if you run out of memory.')

        parser.add_argument('--evaluation_start_epoch', type=int, default=0,
                            help='First epoch to calculate metrics')
        parser.add_argument('--evaluation_freq', type=int, default=100000,
                            help='How often to evaluate (in number of samples seen)')
        parser.add_argument('--num_evaluation_samples', type=int, default=1000,
                            help='Number of samples to use for computing metrics')
        parser.add_argument('--evaluate_val_set', action='store_true',
                            help='Calculate metrics for validation set, too')

        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        parser.add_argument('--lr', type=float, default=0.0002,
                            help='initial learning rate')
        parser.add_argument('--D_steps_per_G', type=int, default=1,
                            help='number of discriminator iterations per generator iterations.')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=32,
                            help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0,
                            help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0,
                            help='weight for vgg loss')

        parser.add_argument('--no_ganFeat_loss', action='store_true',
                            help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true',
                            help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge',
                            help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale',
                            help='(n_layers|multiscale|image)')
        parser.add_argument('--gradient_clip', type=float, default=-1,
                            help='We clip gradients at this value. Use -1 to disable gradient clipping')
        self.isTrain = True
        return parser
