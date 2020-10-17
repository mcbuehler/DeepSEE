"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.  Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import datetime
import ntpath
import os
import time
import traceback
from collections import OrderedDict

import cv2
import numpy as np
import torch

from util import ownhtml
from util.util import save_image
from . import util

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name,
                                         'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(
                    '================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step, logs={},
                                info=None):
        for key, val in logs.items():
            if key.startswith('image'):
                key = key[6:]
                visuals[key] = val
                if not "last" in key:
                    visuals.move_to_end(key,
                                        last=False)  # We want these entries to be shown first

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals, self.opt.batchSize,
                                                self.opt.label_nc)

        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir,
                                                'epoch%.3d_iter%.3d_%s_%d.png' % (
                                                epoch, step, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir,
                                            'epoch%.3d_iter%.3d_%s.png' % (
                                            epoch, step, label))
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]
                    try:
                        util.save_image(image_numpy, img_path)
                    except TypeError:
                        print("Could not save image {} {} {}".format(label,
                                                                     image_numpy.shape,
                                                                     image_numpy.dtype))
                        exit()

            # update website
            webpage = ownhtml.OwnHTML(self.web_dir,
                                      'Experiment name = %s' % self.name,
                                      refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_iter%.3d_%s_%d.png' % (
                            n, step, label, i)
                            ims.append(img_path)
                            txts.append(label + str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_iter%.3d_%s.png' % (
                        n, step, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, height=self.win_size,
                                       info=info)
                else:
                    num = int(round(len(ims) / 2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num],
                                       height=self.win_size, info=info)
                    webpage.add_images(ims[num:], txts[num:], links[num:],
                                       height=self.win_size, info=info)
            webpage.save()

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, total_time_so_far,
                             total_steps_so_far):
        message = '(%s, epoch: %d, iters: %d, time: %.3f, total time: %s, total steps: %s, steps/sec: %d ' % (
        time.strftime("%c"), epoch, i, t,
        str(datetime.timedelta(seconds=total_time_so_far)), total_steps_so_far,
        round(total_steps_so_far / total_time_so_far, 2))
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    @staticmethod
    def convert_visuals_to_numpy(visuals, batchSize, label_nc=None,
                                 dont_tile=False):
        # Make sure to detach them!
        visuals_filtered = OrderedDict()
        for k in visuals:
            if isinstance(visuals[k], torch.Tensor):
                visuals_filtered[k] = visuals[k].clone().detach().cpu()

        visuals = visuals_filtered
        for key, t in visuals.items():
            t = t.detach()
            tile = (not dont_tile and batchSize >= 4)
            if 'input_semantics' in key or 'label' in key:
                t = util.tensor2label(t, label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, info=None):
        visuals = self.convert_visuals_to_numpy(visuals, self.opt.batchSize,
                                                self.opt.label_nc)

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txt = label
            txts.append(txt)
            links.append(image_name)
        webpage.add_images(ims, txts, links, height=self.win_size, info=info)


def save_images_only(visuals, paths, folder_out):
    bs = visuals['image_hr'].size(0)
    label_nc = visuals['input_semantics'].size(1)

    save_keys = ['input_semantics', 'image_lr', 'fake_image', 'image_hr']
    use_guiding_image = "guiding_image" in visuals.keys()
    if use_guiding_image:
        save_keys += ["guiding_image", "guiding_input_label"]

    visuals = OrderedDict([(k, visuals[k].detach().cpu()) for k in save_keys])
    visuals = Visualizer.convert_visuals_to_numpy(visuals, bs, label_nc,
                                                  dont_tile=True)
    for b in range(bs):
        for key in save_keys:
            visual_path = os.path.join(folder_out, key)
            sample_name = paths[b].split('/')[-1]
            save_path = os.path.join(visual_path, sample_name)
            save_image(visuals[key][b], save_path, create_dir=True)
        combined = [
            cv2.resize(visuals['input_semantics'][b],
                       visuals['fake_image'][b].shape[:2]),
            cv2.resize(visuals['image_lr'][b],
                       visuals['fake_image'][b].shape[:2]),
            visuals['fake_image'][b],
            visuals['image_hr'][b]
        ]
        if use_guiding_image:
            combined += [
                visuals['guiding_image'][b],
                cv2.resize(visuals['guiding_input_label'][b],
                           visuals['fake_image'][b].shape[:2])
            ]
        combined = np.concatenate(combined, -2)
        save_path_combined = os.path.join(folder_out, 'combined', sample_name)
        save_image(combined, save_path_combined, create_dir=True)
