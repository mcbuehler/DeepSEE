import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch

import data
from evaluator.PerceptualSimilarity.models.lpips_loss import PerceptualLoss
from evaluator.calculate_PSNR_SSIM import calculate_psnr, calculate_ssim
from evaluator.ssim import msssim
from util.util import tensor2im


class MetricsEvaluator:
    """
    The MetricsEvaluator is resonsible for collecting the scores.
    It optionally writes the results for each sample to an output file.
    """
    psnr_buffer = None
    ssim_buffer = None
    ms_ssim_buffer = None
    rmse_buffer = None
    n_samples = 0
    columns = ["ID", "PSNR", "SSIM", "MSSSIM", "RMSE", "LPIPS"]

    def __init__(self, write_details=False, folder_out=None, cuda=False,
                 extra_columns=(), extra_columns_content=(), append=False):
        """

        Args:
            write_details: If True, the metrics for each sample are written to "folder out".
            folder_out: Target folder for collecting metrics.
            cuda: If True uses GPU
            extra_columns:
            extra_columns_content:
            append:
        """
        assert len(extra_columns) == len(
            extra_columns_content), "Extra columns and content need to be of the same size"
        self.clear()  # This initializes the buffers
        self.cuda = cuda
        self.mse_module = torch.nn.MSELoss(reduction='none')
        self.lpips = PerceptualLoss(use_gpu=cuda)
        self.write_details = write_details
        if self.write_details:
            self.writer = MetricsWriter(folder_out, self.columns,
                                        extra_columns=extra_columns,
                                        extra_columns_content=extra_columns_content,
                                        append=append)
            print("Writing metrics output to {}".format(folder_out))

        if self.cuda:
            self.mse_module = self.mse_module.cuda()

    def clear(self):
        """
        Resets metric buffers
        """
        self.psnr_buffer = []
        self.ssim_buffer = []
        self.ms_ssim_buffer = []
        self.rmse_buffer = []
        self.lpips_buffer = []
        self.n_samples = 0

    def _get_id_from_path(self, path):
        """
        Args:
            path: e.g. ../image/7394.jpg

        Returns: the file_id without extension, e.g. 7394

        """
        return os.path.basename(path)[:-4]

    def _to255(self, tensor):
        """
        Converts a tensor in range [-1, 1] to [0, 255]
        Args:
            tensor: range [-1, 1]

        Returns: Tensor in range [0, 255]

        """
        return (tensor + 1.0) * 127.5

    def collect_samples(self, fake, real, name=None):
        """
        Computes metrics for one batch
        Args:
            fake: Batch of fake images in range [-1, 1]. Shape: (BS,C,H,W)
            real: Batch of fake images in range [-1, 1]. Shape: (BS,C,H,W)
            name: The filename used as an identifier when writing sample results.
                (only required when self.write_details is True)

        Returns: None
        """
        assert fake.size(0) == real.size(0)
        if self.cuda:
            fake = fake.cuda()
            real = real.cuda()
        else:
            fake = fake.detach().cpu()
            real = real.detach().cpu()

        # Computing root mean squared error
        # We compute this on the range [-1, 1]
        rmse = list(self.mse_module(fake, real).mean(
            dim=[1, 2, 3]).sqrt().detach().cpu().numpy())
        self.rmse_buffer += rmse

        # LPIPS
        fake255, real255 = self._to255(fake), self._to255(real)
        lpips = list(self.lpips(fake, real).squeeze(0).squeeze(0).squeeze(
            0).detach().cpu().numpy())
        self.lpips_buffer += lpips

        fake_np = tensor2im(fake)
        real_np = tensor2im(real)
        # We compute these metrics on single samples, hence the for loop
        for i in range(fake.size(0)):
            psnr = calculate_psnr(fake_np[i], real_np[i])
            ssim = calculate_ssim(fake_np[i], real_np[i])
            ms_ssim = msssim(fake255[i].unsqueeze(0), real255[i].unsqueeze(0),
                             size_average=True,
                             val_range=255).detach().cpu().numpy()
            self.ms_ssim_buffer.append(ms_ssim)
            self.psnr_buffer.append(psnr)
            self.ssim_buffer.append(ssim)

            if self.write_details:
                # We keep track of the results for each sample
                image_id = self._get_id_from_path(name[i])
                self.writer.append_line(
                    [image_id, psnr, ssim, ms_ssim, rmse[i], lpips[i]])
        self.n_samples += fake.size(0)

    def get_result(self):
        """
        Returns: OrderedDict with mean and standard deviation for each metric
        """
        data = [
            ("psnr/mean", np.mean(self.psnr_buffer)),
            ("ssim/mean", np.mean(self.ssim_buffer)),
            ("ms_ssim/mean", np.mean(self.ms_ssim_buffer)),
            ("rmse/mean", np.mean(self.rmse_buffer)),
            ("lpips/mean", np.mean(self.lpips_buffer)),
            ("psnr/std", np.std(self.psnr_buffer)),
            ("ssim/std", np.std(self.ssim_buffer)),
            ("ms_ssim/std", np.std(self.ms_ssim_buffer)),
            ("rmse/std", np.std(self.rmse_buffer)),
            ("lpips/std", np.std(self.lpips_buffer)),
            ("n_samples", self.n_samples)
        ]
        return OrderedDict(data)


class MetricsWriter:
    file = None

    def __init__(self, path, metrics, extra_columns=(),
                 extra_columns_content=(), append=False):
        """
        Writes metrics to a file named "metrics.csv"
        Args:
            path: Where to store results CSV
            metrics: Metrics that are saved.
            extra_columns: Extra column headers that should be saved
                (e.g. identifier for dataset or split)
            extra_columns_content: What to write into the extra columns
            append: If True, we append to the metrics.csv file (if it exists)
        """

        self.path_out = os.path.join(path, "metrics.csv")
        self.metrics = metrics
        header = list(extra_columns) + metrics

        if append:
            append_header = not os.path.exists(self.path_out)
            self.file = open(self.path_out, "a")
            if append_header:
                self.append_line(header, add_extra_content=False)
        else:
            self.file = open(self.path_out, "w")
            self.append_line(header, add_extra_content=False)
        self.extra_columns_content = list(extra_columns_content)

    def append_line(self, row, add_extra_content=True):
        content = list(row)
        if add_extra_content:
            content = self.extra_columns_content + content
        self.file.write(",".join(map(str, content)))
        self.file.write(os.linesep)
        self.file.flush()

    def __del__(self):
        if self.file:
            self.file.close()


def get_validation_dataloader(opt):
    """
    Creates a dataloader for the validation set.
    Requires arguments for image_dir_val and label_dir_val
    Args:
        opt: Arguments

    Returns: Dataloader for the valdation set
    """
    assert opt.label_dir_val
    assert opt.image_dir_val
    opt_val = deepcopy(opt)
    opt_val.label_dir = opt.label_dir_val
    opt_val.image_dir = opt.image_dir_val
    dataloader_val = data.create_dataloader(opt_val)
    return dataloader_val


def evaluate_validation_set(inference_manager, model, opt):
    """
    Runs evaluation on the validation set.

    Args:
        inference_manager: inference manager
        model: Model that runs inference
        opt: Arguments

    Returns: Dict with computed metrics
    """
    dataloader_val = get_validation_dataloader(opt)
    print("Evaluating on {} validation samples...".format(
        inference_manager.num_evaluation_samples))
    try:
        result = inference_manager.run(model, dataloader_val)
    except StopIteration:
        # The iterator has exhausted, we log a default value
        print(
            "[!Exception] Running into a StopIteration when calculating validation FID. Logging default FID 500 and continuing...")
        result = {"FID": 500}
    return result


def evaluate_training_set(inference_manager, model, dataloader):
    """
    Runs evaluation on the training set.

    Args:
        inference_manager: inference manager
        model: Model that runs inference

    Returns: Dict with computed metrics
    """
    print("Evaluating on {} training samples...".format(inference_manager.num_samples))
    try:
        result = inference_manager.run(model, dataloader)
    except StopIteration:
        # The iterator has exhausted, we log a default value
        print(
            "[!Exception] Running into a StopIteration when calculating validation FID. Logging default FID 500 and continuing...")
        result = {"FID": 500}
    return result


def inference_validation(model, inference_manager, opt):
    """
    Runs a single inference step on the validation set
    Args:
        model: SRModel
        inference_manager: InferenceManager
        opt: configuration

    Returns: OrderedDict

    """
    dataloader = get_validation_dataloader(opt)

    dataloader = iter(dataloader)
    model = model.eval()

    data_i = next(dataloader)
    out = inference_manager.run_batch(data_i, model)
    model.train()
    out = [("val_{}".format(name), e) for name, e in out.items()]
    return out