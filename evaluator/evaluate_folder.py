
import argparse
import glob
import os

import PIL.Image as Image
import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_msssim import SSIM
from tqdm import tqdm

from evaluator.PerceptualSimilarity.models.lpips_loss import PerceptualLoss
from evaluator.inception_util import get_batch_activations, get_inception_model, \
    calculate_statistics_from_act
from evaluator.pytorch_fid.fid_score import calculate_frechet_distance
from evaluator.ssim import msssim, ssim
from evaluator.psnr import calculate_psnr
from util.util import get_time_string, mkdirs


class MetricsEvaluator:
    METRICS = ["PSNR", "SSIM", "MS-SSIM", "RMSE", "LPIPS", "FID"]
    ssim_module = None
    lpips_module = None
    fid_module = None

    def __init__(self, opt):
        self.ssim_module = SSIM(data_range=255, size_average=True, channel=3)

        self.cuda = opt.cuda
        self.fid_module = get_inception_model().eval()

        if self.cuda:
            print("Using CUDA")
            self.fid_module = self.fid_module.cuda()

        self.lpips_module = PerceptualLoss(use_gpu=self.cuda)

        self.fid_features_real = list()
        self.fid_features_fake = list()

    def _np2tensor(self, np_img):
        assert len(np_img.shape) == 3
        assert np_img.shape[2] == 3
        return torch.from_numpy(np_img.transpose(2, 0, 1)).unsqueeze(0).float()

    def center(self, img):
        tensor = self._np2tensor(img)
        tensor = tensor / 127.5 - 1
        assert -1 <= tensor.min() and tensor.max() <= 1
        return tensor

    def ssim(self, img1, img2):
        img1_tensor = self._np2tensor(img1)
        img2_tensor = self._np2tensor(img2)
        # ssim_value = self.ssim_module(img1_tensor, img2_tensor).item()
        ssim_value = ssim(img1_tensor, img2_tensor).item()
        return ssim_value

    def ms_ssim(self, img1, img2):
        img1_tensor = self._np2tensor(img1)
        img2_tensor = self._np2tensor(img2)
        msssim_value = msssim(img1_tensor, img2_tensor).item()
        return msssim_value

    def psnr(self, img1, img2):
        # img1 and img2 have range [0, 255]
        return calculate_psnr(img1, img2)

    def rmse(self, img1, img2):
        return np.sqrt(np.square(img1 - img2).mean())

    def lpips(self, img1, img2):
        img1_tensor = self.center(img1).float()
        img2_tensor = self.center(img2).float()
        if self.cuda:
            img1_tensor, img2_tensor = img1_tensor.cuda(), img2_tensor.cuda()
        lpips_value = self.lpips_module(img1_tensor, img2_tensor).item()
        return lpips_value

    def collect_fid_statistics(self, img_fake, img_real):
        img_fake_tensor = self.center(img_fake).float()
        img_real_tensor = self.center(img_real).float()
        if self.cuda:
            img_fake_tensor, img_real_tensor = img_fake_tensor.cuda(), img_real_tensor.cuda()
        self.fid_features_fake.append(
            get_batch_activations(self.fid_module, img_fake_tensor))
        self.fid_features_real.append(
            get_batch_activations(self.fid_module, img_real_tensor))
        return -1

    def fid(self):
        all_features_fake = np.concatenate(self.fid_features_fake, 0)
        all_features_real = np.concatenate(self.fid_features_real, 0)
        mu_gen, sigma_gen = calculate_statistics_from_act(all_features_fake)
        mu_real, sigma_real = calculate_statistics_from_act(all_features_real)
        fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_real,
                                         sigma_real)
        return fid

    @classmethod
    def get_available_metrics(cls):
        return cls.METRICS

    def calculate(self, img1, img2, metric_name):
        if metric_name == "PSNR":
            return self.psnr(img1, img2)
        elif metric_name == "SSIM":
            return self.ssim(img1, img2)
        elif metric_name == "MS-SSIM":
            return self.ms_ssim(img1, img2)
        elif metric_name == "RMSE":
            return self.rmse(img1, img2)
        elif metric_name == "LPIPS":
            return self.lpips(img1, img2)
        elif metric_name == "FID":
            return self.collect_fid_statistics(img1, img2)
        else:
            raise ValueError("Unsupported metric: {}".format(metric_name))


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path_real", type=str, required=True)
    parser.add_argument("--path_fake", type=str, required=True)
    parser.add_argument("--results_folder", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=MetricsEvaluator.get_available_metrics(),
                        required=False)
    parser.add_argument("--how_many", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default=-1, choices=(
    "celeba", "bicubic", "celebamaskhq"))
    parser.add_argument("--cuda", action='store_true')

    return parser


def get_image_id(path):
    return path.split("/")[-1][:-4]


def center_crop(img, output_size):
    w, h = img.shape[:2]
    th, tw = output_size
    i = max(0, int(round((h - th) / 2.)))
    j = max(0, int(round((w - tw) / 2.)))
    return img[j:j + th, i:i + tw]


def load_image(path, ):
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float)
    assert 0 <= np.min(img) and np.max(img) <= 255
    return img


def run_evaluation(opt):
    print("==== Running evaluation")
    print("Options:")
    print(opt)
    files = glob.glob(os.path.join(opt.path_fake, "*.png"))
    files.extend(glob.glob(os.path.join(opt.path_fake, '*.jpg')))
    files = sorted(files)
    if len(files) == 0:
        print("No files found. Please make sure that the fake images are in this folder and end with *.png or *.jpg: {}".format(opt.path_fake))
        exit(-1)

    print("{} files found".format(len(files)))

    if opt.how_many > 0:
        files = files[:opt.how_many]

    evaluator = MetricsEvaluator(opt)

    df_metrics = pd.DataFrame(
        columns=['ID'] + list(evaluator.get_available_metrics()))
    for i in tqdm(range(len(files))):
        path_img_fake = files[i]
        image_id = get_image_id(path_img_fake)

        if os.path.exists(
                os.path.join(opt.path_real, "{}.jpg".format(image_id))):
            img_real = load_image(
                os.path.join(opt.path_real, "{}.jpg".format(image_id)))
        elif os.path.exists(
                os.path.join(opt.path_real, "{}.png".format(image_id))):
            img_real = load_image(
                os.path.join(opt.path_real, "{}.png".format(image_id)))
        else:
            raise FileNotFoundError(
                "File id not found (checked jpg and png): {}".format(
                    os.path.join(opt.path_real, image_id)))

        img_fake = load_image(path_img_fake)

        if opt.dataset == 'celeba':
            img_real = center_crop(img_real, (178, 178))
            if img_real.shape != img_fake.shape:
                img_real = cv2.resize(img_real, img_fake.shape[:2],
                                      cv2.INTER_CUBIC).clip(0, 255)
        elif opt.dataset in ['celebamaskhq']:
            if img_real.shape != img_fake.shape:
                img_real = cv2.resize(img_real, img_fake.shape[:2],
                                      cv2.INTER_CUBIC).clip(0, 255)
        else:
            raise ValueError("Unsupported dataset: ", opt.dataset)

        row = {"ID": image_id}
        for key in opt.metrics:
            # print(img_real.shape, img_fake.shape)
            value = evaluator.calculate(img_fake, img_real, key)
            row[key] = value
        df_metrics = df_metrics.append(row, ignore_index=True)

    df_metrics.FID = evaluator.fid()

    path_out = os.path.join(opt.results_folder,
                            "{}_results.csv".format(get_time_string()))
    mkdirs(opt.results_folder)

    df_metrics.to_csv(path_out)
    return path_out


def display_result(path_results):
    df_results = pd.read_csv(path_results, index_col='ID')
    for col in df_results:
        print(col, "Mean: ", df_results[col].mean(), "std:",
              df_results[col].std())


if __name__ == "__main__":
    parser = get_args()
    opt = parser.parse_args()
    path_results = run_evaluation(opt)
    print("Written results to {}".format(path_results))
    display_result(path_results)
