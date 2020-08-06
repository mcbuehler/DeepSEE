"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved. Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy
import random

from PIL import Image

from data.base_dataset import get_params, get_transform, BaseDataset
from data.custom_exception import SkipSampleException


class CelebADataset(BaseDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """
    def initialize(self, opt):
        super().initialize(opt)
        if opt.guiding_style_image:
            assert opt.identities_file, "Please provide an identity file."
            self.id2identity = {}  # maps from file ids to identities
            with open(opt.identities_file, 'r') as f:
                for row in f:
                    filename, identity = row.split(' ')
                    file_id = filename[:-4]
                    self.id2identity[file_id] = identity
            identities = set(self.id2identity.values())
            # maps from identities to a set of file_ids
            self.identity2id = {identity: set() for identity in identities}
            for file_id, identity in self.id2identity.items():
                self.identity2id[identity].add(file_id)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='center_crop_and_resize')
        parser.set_defaults(center_crop_size=178)
        parser.set_defaults(load_size=128)
        parser.set_defaults(crop_size=128)
        parser.set_defaults(start_size=16)
        return parser

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)

        # The labels have already been predicted on square images, so we
        # should not center crop those.
        label_opt = copy.deepcopy(self.opt)
        label_opt.preprocess_mode = 'resize'
        transform_label = get_transform(label_opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        if not self.opt.ignore_path_match:
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
        # We add the guiding image, if required
        input_dict = self.postprocess(input_dict, transform_image, transform_label)
        return input_dict

    def postprocess(self, input_dict, transform_image=None, transform_label=None):
        if self.opt.guiding_style_image:
            input_dict['guiding_image_id'] = self.sample_guiding_image(input_dict['path'])
            input_dict = self.load_guiding(input_dict, transform_image, transform_label)
        return input_dict

    def sample_guiding_image(self, path):
        file_id = path.split('/')[-1][:-4]
        identity = self.id2identity[file_id]
        candidates = set(self.identity2id[identity])
        if self.opt.phase == "test":
            candidates.remove(file_id)
            if len(candidates) == 0:
                guiding_image_id = None
                raise SkipSampleException("There is no other candidate for file id: {}".format(file_id))
        guiding_image_id = random.sample(candidates, 1)[0]
        return guiding_image_id
