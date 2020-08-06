import csv
import os
import random

import pandas as pd

from .base_dataset import BaseDataset


class CelebAMaskHQDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        return parser

    def initialize(self, opt):
        super().initialize(opt)

        if opt.guiding_style_image:
            assert opt.identities_file, "Please provide an identity file."
            assert os.path.exists(opt.identities_file), "Please provide a correct path to the identities file (invalid path: {})".format(opt.identities_file)
            self.df_identities = pd.read_csv(opt.identities_file, quoting=csv.QUOTE_ALL, dtype=str, index_col=0)

            # We filter for the given dataset split
            file_ids = [p.split('/')[-1][:-4] for p in self.image_paths]
            self.df_identities = self.df_identities.loc[self.df_identities['hq_file_id'].isin(file_ids)]

    def postprocess(self, input_dict, transform_image=None, transform_label=None):
        if self.opt.guiding_style_image:
            input_dict['guiding_image_id'] = self.sample_guiding_image(input_dict['path'])
            input_dict = self.load_guiding(input_dict, transform_image,
                                           transform_label)
        return input_dict

    def sample_guiding_image(self, path):
        file_id = path.split('/')[-1][:-4]
        identity = self.df_identities[self.df_identities['hq_file_id']==file_id]['identity'].values[0]
        candidates = set(self.df_identities[self.df_identities['identity']==identity]['hq_file_id'].values)
        if self.opt.phase != "train":
            # We do want to make sure to use a different image to the real image.
            candidates.remove(file_id)
        guiding_image_id = random.sample(candidates, 1)[0]
        return guiding_image_id


