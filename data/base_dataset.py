"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.  Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import random

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from util import util
from data.image_folder import make_dataset


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        parser.set_defaults(preprocess_mode='resize_and_crop')

        parser.add_argument('--label_dir', type=str,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str,
                            help='path to the directory that contains photo images')

        parser.add_argument('--label_dir_val', type=str, required=False,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir_val', type=str, required=False,
                            help='path to the directory that contains photo images')

        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def initialize(self, opt):
        self.opt = opt

        self.downsampling_method = Image.BICUBIC
        if self.opt.downsampling_method == "bilinear":
            self.downsampling_method = Image.BILINEAR

        label_paths, image_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/base_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        print("image dir: {}".format(opt.image_dir))
        print("label dir: {}".format(opt.label_dir))
        if not opt.no_pairing_check:
            assert len(label_paths) == len(image_paths), "The #images ({}) in {} and ({}) in {} do not match. Is there something wrong?".format(len(label_paths), label_dir, len(image_paths), image_dir)

        return label_paths, image_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)

        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), \
                "The label_path %s and image_path %s don't match." % \
                (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params, method=self.downsampling_method)
        image_tensor = transform_image(image)

        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        input_dict = self.postprocess(input_dict, transform_image, transform_label)
        return input_dict

    def load_guiding(self, input_dict, transform_image, transform_label):
        """
        This loads and preprocesses a high-resolution reference image.
        Args:
            input_dict: dict, should contain a guiding_image_id
            transform_image: transform to apply to the image
            transform_label: transform to apply to the label

        Returns: input_dict with added guided image, label

        """
        guiding_image_path = os.path.join(self.opt.image_dir, "{}.jpg".format(
            input_dict['guiding_image_id']))
        guiding_image = Image.open(guiding_image_path).convert('RGB')
        guiding_image_tensor = transform_image(guiding_image)
        input_dict['guiding_image'] = guiding_image_tensor

        guiding_label_path = os.path.join(self.opt.label_dir, "{}.png".format(
            input_dict['guiding_image_id']))
        guiding_label = Image.open(guiding_label_path)
        guiding_label_tensor = transform_label(guiding_label) * 255.0
        input_dict['guiding_label'] = guiding_label_tensor
        return input_dict

    def postprocess(self, input_dict, transform_image=None, transform_label=None):
        return input_dict

    def __len__(self):
        return self.dataset_size


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True, preprocess_mode=None):
    transform_list = []
    preprocess_mode = opt.preprocess_mode if preprocess_mode is None else preprocess_mode
    if 'center_crop' in preprocess_mode:
        transform_list.append(transforms.CenterCrop(opt.center_crop_size))
    if 'resize' in preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    elif 'scale_shortside' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, method)))

    if 'crop' in preprocess_mode and not 'center_crop' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
