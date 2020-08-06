"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.  Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import datetime
import os
import time

import numpy as np


# Helper class that keeps track of training iterations
class IterationCounter():
    def __init__(self, opt, dataset_size):
        self.opt = opt
        self.dataset_size = dataset_size

        self.first_epoch = 1
        self.total_epochs = opt.niter + opt.niter_decay
        self.epoch_iter = 0  # iter number within each epoch
        self.total_time_so_far = 0
        self.iter_record_path = os.path.join(self.opt.checkpoints_dir,
                                             self.opt.name, 'iter.txt')
        self.fid_record_path = os.path.join(self.opt.checkpoints_dir,
                                            self.opt.name, 'fid_iter.txt')
        self.metrics_record_path = os.path.join(self.opt.checkpoints_dir,
                                                self.opt.name,
                                                'metrics_iter.txt')
        # Used to keep track of iters at beginning of epoch when reloading
        self.keep_last_iter = False
        if opt.isTrain and opt.continue_train:
            self.keep_last_iter = True
            try:
                self.first_epoch, self.epoch_iter, self.total_time_so_far = np.loadtxt(
                    self.iter_record_path, delimiter=',', dtype=int)
                if self.opt.which_epoch != "latest":
                    self.first_epoch = int(self.opt.which_epoch)
                    self.epoch_iter = 0
                print('Resuming from epoch %d at iteration %d' % (
                self.first_epoch, self.epoch_iter))
            except Exception as e:
                print(e)
                print(
                    'Could not load iteration record at %s. Starting from beginning.' %
                    self.iter_record_path)
        self.total_steps_so_far = (
                                  self.first_epoch - 1) * dataset_size + self.epoch_iter

    # return the iterator of epochs for the training
    def training_epochs(self):
        return range(self.first_epoch, self.total_epochs + 1)

    def record_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        if not self.keep_last_iter:
            self.epoch_iter = 0
        self.keep_last_iter = False
        self.last_iter_time = time.time()
        self.current_epoch = epoch

    def record_one_iteration(self):
        current_time = time.time()
        time_for_iter = current_time - self.last_iter_time
        # the last remaining batch is dropped (see data/__init__.py),
        # so we can assume batch size is always opt.batchSize
        self.total_time_so_far = self.total_time_so_far + time_for_iter
        self.time_per_iter = time_for_iter / self.opt.batchSize
        self.last_iter_time = current_time
        self.total_steps_so_far += self.opt.batchSize  # total  steps so far is actually the number of seen images
        self.epoch_iter += self.opt.batchSize

    def print_iter_info(self):
        time_string = str(datetime.timedelta(seconds=self.total_time_so_far))
        print(
            'Saved current iteration count ({}) at {}. Total time so far: {}.'.format(
                self.total_steps_so_far, self.iter_record_path, time_string))

    def record_epoch_end(self):
        current_time = time.time()
        self.time_per_epoch = current_time - self.epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (self.current_epoch, self.total_epochs, self.time_per_epoch))
        if self.current_epoch % self.opt.save_epoch_freq == 0:
            np.savetxt(self.iter_record_path,
                       (self.current_epoch + 1, 0, self.total_time_so_far),
                       delimiter=',', fmt='%d')
            self.print_iter_info()

    def record_current_iter(self):
        np.savetxt(self.iter_record_path, (
        self.current_epoch, self.epoch_iter, self.total_time_so_far),
                   delimiter=',', fmt='%d')
        self.print_iter_info()

    def get_time_string(self):
        time_string = str(datetime.datetime.strftime(datetime.datetime.now(),
                                                     '%Y/%m/%d-%H:%M:%S'))
        return time_string

    def record_fid(self, fid, split, num_samples):
        with open(self.fid_record_path, "a") as f:
            msg = "time={},split={},num_samples={},epoch={:03d},total_steps_so_far={:010d},fid={}".format(
                self.get_time_string(), split, num_samples, self.current_epoch,
                self.total_steps_so_far, fid)
            f.write(msg)
            f.write(os.linesep)
            print(msg)
        return msg

    def record_metrics(self, metrics_dict, split):
        with open(self.metrics_record_path, "a") as f:
            msg = "time={},split={},num_samples={},epoch={:03d},total_steps_so_far={:010d},".format(
                self.get_time_string(), split, metrics_dict["n_samples"],
                self.current_epoch, self.total_steps_so_far)
            msg += ",".join(
                ["{}={}".format(k, v) for k, v in metrics_dict.items() if
                 "psnr" in k or "ssim" in k or "rmse" in k])
            f.write(msg)
            f.write(os.linesep)
            print(msg)
        return msg

    def needs_saving(self):
        return (
               self.total_steps_so_far % self.opt.save_latest_freq) < self.opt.batchSize

    def needs_printing(self):
        return (
               self.total_steps_so_far % self.opt.print_freq) < self.opt.batchSize

    def needs_displaying(self):
        return (
               self.total_steps_so_far % self.opt.display_freq) <= self.opt.batchSize

    def needs_evaluation(self):
        return self.current_epoch + 1 >= self.opt.evaluation_start_epoch and \
               (
               self.total_steps_so_far % self.opt.evaluation_freq) < self.opt.batchSize
