"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.  Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import datetime
import importlib
import os
import re
from collections import OrderedDict

import dill as pickle
import numpy as np
import torch
from PIL import Image


# import util.coco


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# returns a configuration for creating a generator
# |default_opt| should be the opt of the current experiment
# |**kwargs|: if any configuration should be overriden, it can be specified here


def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate(
            [imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)],
            axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(
            np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)],
                           axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False,
                 picturesPerRow=4):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np,
                                       picturesPerRow=picturesPerRow)
            return images_tiled
        else:
            # images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)
    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))


def save_style_matrix(tensor, path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    assert len(tensor.shape) == 2, "Shape is incorrect: {}".format(
        tensor.shape)
    assert path.endswith(".csv")
    # save to csv
    style_np = np.array(tensor.detach().cpu())
    np.savetxt(os.path.join(path), style_np, delimiter=',')


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print(
            "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (
            module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    checkpoint = {
        "model": net.cpu().state_dict(),
    }
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(checkpoint, save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    checkpoint = torch.load(save_path)
    if "model" in checkpoint:
        net.load_state_dict(checkpoint["model"])
    else:
        net.load_state_dict(checkpoint)
    return net


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                         (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160),
                         (230, 150, 140), (70, 70, 70), (102, 102, 156),
                         (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90),
                         (153, 153, 153), (153, 153, 153), (250, 170, 30),
                         (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180),
                         (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100),
                         (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        if N == 182:  # COCO
            important_colors = {
                'sea': (54, 62, 167),
                'sky-other': (95, 219, 255),
                'tree': (140, 104, 47),
                'clouds': (170, 170, 170),
                'grass': (29, 195, 49)
            }
            for i in range(N):
                name = util.coco.id2label(i)
                if name in important_colors:
                    color = important_colors[name]
                    cmap[i] = np.array(list(color))

    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
        return color_image


def create_mixed_z(batch_size, z_dim, number_regions, n_common=None,
                   region_indices_common=None, n_same_samples=2):
    """
    e.g. sample_mixed_z(2, 16, 4, 2)
    Args:
        batch_size:
        z_dim:
        n_regions:
        n_common:

    Returns:

    """
    assert batch_size % 2 == 0, "We need to create pairs, so batch size should be an even number"
    assert z_dim % number_regions == 0
    if n_common:
        assert n_common < number_regions, "You wanted to have {} common regions, but there are only {} regions. " \
                                          "We need to have fewer common regions than there are regions".format(
            n_common, number_regions)
    if region_indices_common is None:
        # We sample the region indices
        region_indices_common = torch.from_numpy(
            np.random.choice(number_regions, size=(1, n_common),
                             replace=False))
    else:
        region_indices_common = [region_indices_common]
    sample_indices_common = torch.tensor(
        list(range(0, batch_size, n_same_samples)))
    # For each region we have n_per_region elements uniformly distributed in [-1, 1)
    n_per_region = z_dim // number_regions
    # torch.rand samples from [0, 1)
    # print(region_indices_common)
    z = torch.rand((batch_size, number_regions, n_per_region)) * 2 - 1
    for sample_idx in sample_indices_common:
        to_copy = z[sample_idx, region_indices_common]
        z[sample_idx:sample_idx + n_same_samples,
        region_indices_common] = to_copy  # TODO: why does it work here?
        # print(z[sample_idx:sample_idx+2,:6,0])

    # to_copy = z[sample_indices_common + torch.ones_like(sample_indices_common), region_indices_common]
    # z[sample_indices_common, region_indices_common] = to_copy
    return z, sample_indices_common, region_indices_common


def get_celebA_regions():
    return [
        "Background",  # 0
        "Skin",
        "Nose",
        "Eyeglass",
        "Left eye",
        "Right eye",  # 5
        "Left eyebrow",
        "Right eyebrow",
        "Left Ear",
        "Right Ear",
        "Mouth",  # 10
        "Upper Lip",
        "Lower Lip",
        "Hair",  # 13
        "Hat",
        "Earring",  # 15
        "Necklace",
        "Neck",
        "Cloth"]


# 3 4 5 6 7 11 12 13 15 18


def get_celebA_region_name(index, nospace=False):
    labels = get_celebA_regions()
    assert index < len(labels)
    name = get_celebA_regions()[index]
    if nospace:
        name = name.replace(' ', '')
    return name


def get_celebA_region_index(names):
    if isinstance(names, list):
        return [get_celebA_region_index(n) for n in names]
    index = get_celebA_regions().index(names)
    assert index >= 0
    return index


def display_regions():
    for region_id in range(19):
        print("{:2d}: {}".format(region_id, get_celebA_region_name(region_id)))


def print_gpu_memory():
    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    out_str = sp.communicate()
    for e in out_str[0].decode('utf-8').split('\n'):
        if "GPU Memory" in e:
            print(e)


def tensor_hwc2ch2(tensor):
    assert len(tensor.shape) == 3
    return tensor.transpose(0, 1).transpose(0, 2).contiguous()


def get_time_string(format="%Y%m%d_%H%M"):
    time_str = datetime.datetime.now().strftime(format)
    return time_str


def gpu_info(message, opt):
    if opt.gpu_info and torch.cuda.is_available():
        import GPUtil
        GPUtil.showUtilization()
        print("-------------------------", message)


def filter_none(ordered_dict):
    out = OrderedDict(
        [(k, v) for k, v in ordered_dict.items() if v is not None])
    return out


class ObjectDict(dict):
    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            setattr(self, k, v)
