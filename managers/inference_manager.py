import datetime
import os
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm

from managers.base_manager import BaseManager
from data.custom_exception import SkipSampleException
from evaluator.evaluation import MetricsEvaluator
from evaluator.pytorch_fid.fid_score import calculate_frechet_distance
from util.util import mkdirs
from util.visualizer import save_images_only
from evaluator.inception_util import calculate_statistics_from_act, get_batch_activations, get_inception_model


class InferenceManager(BaseManager):
    """
    InferenceManager
    """
    def __init__(self, opt, num_samples, write_details=False, folder_out=None,
                 save_images=False, cuda=False):
        super().__init__(opt, create_model=False)
        self.num_samples = num_samples
        self.batch_size = opt.batchSize
        self.write = write_details
        self.save_image = save_images
        self.folder_out = folder_out

        if self.save_image or self.write:
            mkdirs(self.folder_out)

        # The MetricsEvaluator is responsible for collecting the scores.
        # It optionally writes the results for each sample to an output file.

        self.metrics = MetricsEvaluator(write_details, folder_out, cuda=cuda)

        self.cuda = cuda
        if self.cuda:
            print("Using CUDA for evaluator")

    def save_stats(self, mu, sigma, folder, bs, is_real):
        suffix = "real" if is_real else "fake"
        mkdirs(folder)
        output_path = os.path.join(folder,
                                   'fid_stats_{}samples_{}.npz'.format(bs,
                                                                       suffix))
        np.savez_compressed(output_path, mu=mu, sigma=sigma)

    def run_batch(self, data, model):
        data = super().preprocess(data, from_dataloader=True)
        with torch.no_grad():
            out = model(data, "inference")
        return out

    def run(self, model, dataloader):
        dataloader = iter(dataloader)
        model = model.eval()

        start_time = time.time()

        fid_model = get_inception_model().eval()
        if self.cuda:
            fid_model = fid_model.cuda()

        num_batches = self.num_samples // self.batch_size + 1

        skipped_samples = 0

        all_features_fake = []
        all_features_real = []
        for i in tqdm(range(num_batches)):
            if i > 0 and i * self.batch_size % 500 < self.batch_size:
                print("\rCurrent result: {}".format(self.metrics.get_result()))
            try:
                data_i = next(dataloader)
                out = self.run_batch(data_i, model)
                gen_full_images = out['fake_image'].detach()
                real_full_images = out['image_hr'].detach()
                if not self.cuda:
                    gen_full_images = gen_full_images.cpu()
                    real_full_images = real_full_images.cpu()

                self.metrics.collect_samples(gen_full_images, real_full_images,
                                         data_i['path'])
                all_features_fake.append(
                    get_batch_activations(fid_model, batch=gen_full_images))
                all_features_real.append(
                    get_batch_activations(fid_model, batch=real_full_images))

                if self.save_image:
                    save_images_only(out, data_i['path'],
                                     os.path.join(self.folder_out, "visuals"))
            except SkipSampleException:
                print("Skipping sample...")
                skipped_samples += 1
            except ValueError as e:
                print(traceback.format_exc())
                print(sys.exc_info()[0])
                print("Value error. Skipping sample...")
                skipped_samples += 1
            except StopIteration:
                print("StopIteration raised. Finishing up...")
                break

        all_features_fake = np.concatenate(all_features_fake, 0)
        all_features_real = np.concatenate(all_features_real, 0)

        # Calculating FID score. This can take a while.
        mu_gen, sigma_gen = calculate_statistics_from_act(all_features_fake)
        mu_real, sigma_real = calculate_statistics_from_act(all_features_real)
        if self.write:
            print("Writing results to {}...".format(self.folder_out))
            self.save_stats(mu_gen, sigma_gen, self.folder_out,
                            all_features_fake.shape[0], is_real=False)
            self.save_stats(mu_real, sigma_real, self.folder_out,
                            all_features_fake.shape[0], is_real=True)

        try:
            cur_fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_real,
                                                 sigma_real)
        except Exception as e:
            print(e)
            cur_fid = 500

        time_delta = datetime.timedelta(
            seconds=time.time() - start_time)
        print("FID finished. FID: {:3.2f}. Time: {}".format(cur_fid,
                                                                str(
                                                                    time_delta)))

        result = OrderedDict([("FID", cur_fid)])

        # We now add the aggregated scores for other metrics
        result.update(self.metrics.get_result())
        self.metrics.clear()

        model.train()
        print(
            "Evaluation finished. Total number of samples skipped: {}".format(
                skipped_samples))
        return result
