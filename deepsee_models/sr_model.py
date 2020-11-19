"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.  Modified by Marcel Bühler.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import random
from collections import OrderedDict
import numpy as np
import torch
import deepsee_models.networks as networks
import util.util as util
from util.util import gpu_info


class SRModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.use_E = opt.netE is not None and len(opt.netE)
        self.netSR, self.netD, self.netE = self.initialize_networks(opt)
        self.mp = self.opt.model_parallel_mode

        self.model_variant = "guided" if "full" in self.opt.netE else "independent"

        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)

        # holding variables for logging. Is overwritten at every iteration
        self.logs = OrderedDict()

        self.last_encoded_style_is_full = True
        self.last_encoded_style_is_noisy = False
        gpu_info("Init SR Model", self.opt)

    def load_weights(self):
        opt = self.opt
        if not opt.isTrain or opt.continue_train:
            self.netSR = util.load_network(self.netSR, 'SR', opt.which_epoch, opt)
            if opt.isTrain:
                self.netD = util.load_network(self.netD, 'D', opt.which_epoch, opt)
            if self.use_E:
                self.netE = util.load_network(self.netE, 'E', opt.which_epoch, opt)

    def get_logs(self):
        return self.logs

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode, **kwargs):
        if self.opt.model_parallel_mode == 1:
            with torch.cuda.device(1):
                torch.cuda.empty_cache()
        gpu_info("Forward started", self.opt)
        input_semantics = data.get("input_semantics", None)
        image_lr = data.get("image_lr", None)
        image_hr = data.get("image_hr", None)  # Only required if training
        # Only required if netE is "fullstyle"
        guiding_image = data.get("guiding_image", None)
        guiding_label = data.get("guiding_label", None)
        encoded_style = data.get("encoded_style", None)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, image_hr, image_lr, guiding_image, guiding_label)
            self.logs['image/downsized'] = image_lr
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, image_hr, image_lr, guiding_image, guiding_label)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _, _ = self.generate_fake(input_semantics=input_semantics, image_downsized=image_lr, full_image=image_hr, no_noise=True, guiding_image=guiding_image, guiding_label=guiding_label)
            data["fake_image"] = fake_image
            # Filter out None values
            data = util.filter_none(data)
            return data
        elif mode == 'encode_only':
            encoded_style, encoder_activations = self.encode_style(
                downscaled_image=image_lr,
                input_semantics=input_semantics, full_image=image_hr,
                no_noise=True, guiding_image=guiding_image,
                guiding_label=guiding_label,
                encode_full=self.opt.full_style_image)
            return encoded_style
        elif mode == 'demo':
            with torch.no_grad():
                fake_image = self.netSR(image_lr, seg=input_semantics,
                                        z=encoded_style)
            out = data
            out["fake_image"] = fake_image
            # Filter out None values
            out = util.filter_none(out)
            return out
        elif mode == 'baseline':
            image_baseline = networks.F.interpolate(image_lr,
                                   (image_hr.shape[-2:]), mode='bicubic').clamp(-1, 1)
            return OrderedDict([("input_label", input_semantics),
                    ("image_downsized", image_lr),
                    ("fake_image", image_baseline),
                    ("image_full", image_hr)])
        elif mode == "inference_noise":
            with torch.no_grad():
                n = self.opt.batchSize
                image_downsized_repeated = image_lr.repeat_interleave(n, 0)
                input_semantics_repeated = input_semantics.repeat_interleave(n, 0)
                # encoded_style = torch.randn((image_downsized_repeated.size(0), self.opt.label_nc, self.opt.regional_style_size), device=image_downsized_repeated.device) / 100
                fake_image, _, _ = self.generate_fake(input_semantics=input_semantics_repeated, image_downsized=image_downsized_repeated, encoded_style=None)

                fake_image = torch.stack(
                    [fake_image[i * n:i * n +n] for i in range(n)], dim=0)  # Shape bs, bs, 3, H, W
                return OrderedDict([("input_label", input_semantics),
                                    ("image_downsized", image_lr),
                                    ("fake_image", fake_image),
                                    ("image_full", image_hr)])
        elif mode == "inference_multi_modal":
            # Randomly varies the appearance for the given regions
            with torch.no_grad():
                n = self.opt.n_interpolation  # TODO: rename to something else
                consistent_regions = np.array([4, 6, 8, 11])  # TODO: store in dataset class

                encoded_style, _ = self.encode_style(downscaled_image=image_lr,
                                                              input_semantics=input_semantics,
                                                              full_image=image_hr,
                                                              no_noise=True, guiding_image=guiding_image, guiding_label=guiding_label)
                region_idx = self.opt.region_idx if self.opt.region_idx else list(range(input_semantics.size(1)))

                # region_idx = np.random.choice(region_idx, 1)
                delta = self.opt.noise_delta
                fake_images = list()
                applied_style = list()
                for b in range(self.opt.batchSize):
                    fake_samples = list()
                    style_samples = list()
                    for i in range(n):
                        encoded_style_in = encoded_style[b].clone().detach()
                        noise = self.get_noise(encoded_style_in[region_idx].shape, delta)
                        encoded_style_in[region_idx] = (encoded_style_in[region_idx] + noise).clamp(-1, 1)
                        encoded_style_in[consistent_regions] = encoded_style_in[consistent_regions+1]
                        fake_image, _, _ = self.generate_fake(input_semantics=input_semantics[b].unsqueeze(0),
                                                                 image_downsized=image_lr[b].unsqueeze(0),
                                                                 encoded_style=encoded_style_in.unsqueeze(0))
                        fake_samples.append(fake_image)
                        style_samples.append(encoded_style_in)

                    if not self.opt.dont_merge_fake:
                        to_append = torch.cat(fake_samples, -1)
                    else:
                        to_append = torch.stack(fake_samples, 1)
                        to_append_style = torch.stack(style_samples)
                    fake_images.append(to_append)
                    applied_style.append(to_append_style)
            fake_out = torch.cat(fake_images, 0)
        elif mode == "inference_replace_semantics":
            with torch.no_grad():
                # region_idx = self.opt.region_idx if self.opt.region_idx else list(range(input_semantics.size(1)))
                fake_images = list()

                fake_image, _, _ = self.generate_fake(input_semantics=input_semantics,
                                                         image_downsized=image_lr)
                fake_images.append(fake_image)

                regions_replace = [10]
                new_region_idx = 12
                for i, rp in enumerate(regions_replace):
                    if isinstance(new_region_idx, list):
                        data['label'][data['label'] == rp] = new_region_idx[i]
                    else:
                        data['label'][data['label'] == rp] = new_region_idx
                input_semantics, image_hr, image_lr, guiding_image, guiding_label, guiding_image2, guiding_label2 = self.preprocess_input(data)  # guiding image can be None
                fake_image, _, _ = self.generate_fake(input_semantics=input_semantics,
                                                         image_downsized=image_lr)
                fake_images.append(fake_image)
                fake_out = torch.cat(fake_images, -1)
                out = OrderedDict([("input_label", input_semantics),
                             ("image_downsized", image_lr),
                             ("fake_image", fake_out),
                             ("image_full", image_hr)])
                if self.opt.guiding_style_image:
                    out['guiding_image_id'] = data['guiding_image_id']
                    out['guiding_image'] = data['guiding_image']
                    out['guiding_input_label'] = data['guiding_label']
                return out
        elif mode == "inference_reference_semantics":
            with torch.no_grad():
                # region_idx = self.opt.region_idx if self.opt.region_idx else list(range(input_semantics.size(1)))
                fake_images = list()
                bak_input_semantics = input_semantics.clone().detach()
                for b in range(self.opt.batchSize):
                    current_semantics = input_semantics.clone().detach()
                    for b_sem in range(self.opt.batchSize):
                        current_semantics[b] = bak_input_semantics[(b_sem)]
                    fake_image, _, _ = self.generate_fake(input_semantics=current_semantics, image_downsized=image_lr)
                    fake_images.append(fake_image)
                fake_out = torch.cat(fake_images, -1)
                out = OrderedDict([("input_label", input_semantics),
                             ("image_downsized", image_lr),
                             ("fake_image", fake_out),
                             ("image_full", image_hr)])
                if self.opt.guiding_style_image:
                    out['guiding_image_id'] = data['guiding_image_id']
                    out['guiding_image'] = data['guiding_image']
                    out['guiding_input_label'] = data['guiding_label']
                return out
        elif mode == "inference_interpolation":
            with torch.no_grad():
                if "style_matrix" in data:
                    encoded_style = data["style_matrix"]
                else:
                    encoded_style, _ = self.encode_style(downscaled_image=image_lr,
                                                           input_semantics=input_semantics, full_image=image_hr,
                                                           no_noise=True, guiding_image=guiding_image, guiding_label=guiding_label)
                n = self.opt.n_interpolation
                assert n % 2 == 1, "Please use an odd n such that the middle image has delta=0"
                delta = self.opt.noise_delta
                region_idx = self.opt.region_idx if self.opt.region_idx else list(range(input_semantics.size(1)))
                fake_images = list()
                applied_style = list()
                for b in range(self.opt.batchSize):
                    fake_samples = list()
                    style_samples = list()
                    for delta_step in np.linspace(-delta, delta, num=n):
                        encoded_style_in = encoded_style[b].clone().detach()
                        encoded_style_in[region_idx] = (encoded_style_in[region_idx] + delta_step).clamp(-1, 1)
                        fake_image, _, _ = self.generate_fake(input_semantics=input_semantics[b].unsqueeze(0),
                                                                 image_downsized=image_lr[b].unsqueeze(0),
                                                                 encoded_style=encoded_style_in.unsqueeze(0))
                        fake_samples.append(fake_image)
                        style_samples.append(encoded_style_in)
                    if not self.opt.dont_merge_fake:
                        to_append = torch.cat(fake_samples, -1)
                    else:
                        to_append = torch.stack(fake_samples, 1)
                        to_append_style = torch.stack(style_samples)
                        applied_style.append(to_append_style)
                    fake_images.append(to_append)
                fake_out = torch.cat(fake_images, 0)
                out = OrderedDict([("input_label", input_semantics),
                             ("image_downsized", image_lr),
                             ("fake_image", fake_out),
                             ("image_full", image_hr),
                            ("style", applied_style)])
                if self.opt.guiding_style_image:
                    out['guiding_image_id'] = data['guiding_image_id']
                    out['guiding_image'] = data['guiding_image']
                    out['guiding_input_label'] = data['guiding_label']
                return out
        elif mode == "inference_interpolation_style":
            with torch.no_grad():
                encoded_style_from = data["style_from"].to(input_semantics.device)
                encoded_style_to = data["style_to"].to(input_semantics.device)
                n = self.opt.n_interpolation
                assert n % 2 == 1, "Please use an odd n such that the middle image has delta=0"
                fake_images = list()
                applied_style = list()
                for b in range(self.opt.batchSize):
                    fake_samples = list()
                    style_samples = list()
                    for delta_step in np.linspace(0, 1, num=n):
                        encoded_style_in = (1 - delta_step) * encoded_style_from[b].clone().detach() + (delta_step * encoded_style_to[b].clone().detach())
                        fake_image, _, _ = self.generate_fake(input_semantics=input_semantics[b].unsqueeze(0),
                                                                 image_downsized=image_lr[b].unsqueeze(0),
                                                                 encoded_style=encoded_style_in.unsqueeze(0))
                        fake_samples.append(fake_image)
                        style_samples.append(encoded_style_in)
                    if not self.opt.dont_merge_fake:
                        to_append = torch.cat(fake_samples, -1)
                    else:
                        to_append = torch.stack(fake_samples, 1)
                        to_append_style = torch.stack(style_samples)
                        applied_style.append(to_append_style)
                    fake_images.append(to_append)
                fake_out = torch.cat(fake_images, 0)
                out = OrderedDict([("input_label", input_semantics),
                             ("image_downsized", image_lr),
                             ("fake_image", fake_out),
                             ("image_full", image_hr),
                            ("style", applied_style)])
                if self.opt.guiding_style_image:
                    out['guiding_image_id'] = data['guiding_image_id']
                    out['guiding_image'] = data['guiding_image']
                    out['guiding_input_label'] = data['guiding_label']
                return out
        elif mode == "inference_particular_combined":
            with torch.no_grad():
                encoded_style_mini, _ = self.encode_style(input_semantics=input_semantics,
                                                                downscaled_image=image_lr,
                                                                  no_noise=True, encode_full=False,
                                                                  guiding_image=None,
                                                                  guiding_label=None)
                if self.opt.noise_delta > 0:
                    region_idx = self.opt.region_idx if self.opt.region_idx else list(
                        range(input_semantics.size(1)))
                    print("Adding noise to style for regions {}".format(region_idx))

                    noise = self.get_noise(encoded_style_mini[:, region_idx].shape,
                                           self.opt.noise_delta)
                    encoded_style_mini[:, region_idx] = (
                        encoded_style_mini[:, region_idx] + noise).clamp(-1, 1)
                    consistent_regions = np.array(
                        [4, 6, 8, 11])  # TODO: store in dataset class
                    encoded_style_mini[:, consistent_regions] = encoded_style_mini[:,
                        consistent_regions + 1]
                    fake_image, _, _ = self.generate_fake(
                        input_semantics=input_semantics,
                        image_downsized=image_lr,
                        encoded_style=encoded_style_mini)
                else:
                    fake_image, _, _ = self.generate_fake(input_semantics=input_semantics,
                                                             image_downsized=image_lr,
                                                             encoded_style=encoded_style_mini)

                # encoded_style_guided, _, _, _ = self.encode_style(downscaled_image=None,
                #                                                   no_noise=True, encode_full=True,
                #                                                   guiding_image=guiding_image,
                #                                                   guiding_label=guiding_label)
                # encoded_style_modified = encoded_style_mini.clone().detach()
                # encoded_style_modified[0, region_idx] = encoded_style_guided[0, region_idx]
                # fake_image_modified, _, _, _ = self.generate_fake(input_semantics=input_semantics,
                #                                              image_downsized=image_downsized,
                #                                              encoded_style=encoded_style_modified)

                out = OrderedDict([("input_label", input_semantics),
                             ("image_downsized", image_lr),
                             ("fake_image_original", fake_image),
                             # ("fake_image_modified", fake_image_modified),
                             ("image_full", image_hr)])
                if self.opt.guiding_style_image:
                    out['guiding_image_id'] = data['guiding_image_id']
                    out['guiding_image'] = data['guiding_image']
                    out['guiding_input_label'] = data['guiding_label']
                return out
        elif mode == "inference_particular_full":
            with torch.no_grad():

                region_idx = self.opt.region_idx if self.opt.region_idx else list(range(input_semantics.size(1)))
                encoded_style_full, _ = self.encode_style(input_semantics=None,
                                                                downscaled_image=None,
                                                                  no_noise=True, encode_full=True,
                                                                  guiding_image=image_hr,
                                                                  guiding_label=input_semantics)
                fake_image_original, _, _ = self.generate_fake(input_semantics=input_semantics,
                                                             image_downsized=image_lr,
                                                             encoded_style=encoded_style_full)

                out = OrderedDict([("input_label", input_semantics),
                                   ("image_downsized", image_lr),
                                   ("fake_image_original", fake_image_original),
                                   ("image_full", image_hr)])

                if self.opt.guiding_style_image:
                    guiding_style, _ = self.encode_style(input_semantics=None,
                                                                 downscaled_image=None,
                                                                 no_noise=True, encode_full=True,
                                                                 guiding_image=guiding_image,
                                                                 guiding_label=guiding_label)
                    # The fake image produced with a guiding style
                    fake_image_guiding, _, _ = self.generate_fake(input_semantics=input_semantics,
                                                                      image_downsized=image_lr,
                                                                      encoded_style=guiding_style)

                    out["fake_image_guiding"] = fake_image_guiding
                    out['guiding_image_id'] = data['guiding_image_id']
                    out['guiding_image'] = data['guiding_image']
                    out['guiding_input_label'] = data['guiding_label']
                return out
        elif mode == "inference_reference":
            with torch.no_grad():
                # encoded_style_mini, _, _, _ = self.encode_style(downscaled_image=image_downsized,
                #                                            input_semantics=input_semantics, full_image=None,
                #                                            no_noise=True)
                encoded_style_full, _ = self.encode_style(downscaled_image=None,
                                                           input_semantics=input_semantics, full_image=image_hr,
                                                           no_noise=True, encode_full=True, guiding_image=guiding_image, guiding_label=guiding_label)
                region_idx = self.opt.region_idx if self.opt.region_idx else list(range(input_semantics.size(1)))
                fake_images = list()
                for b in range(self.opt.batchSize):
                    fake_samples = list()
                    for semantics_b in range(self.opt.batchSize):
                        encoded_style_in = encoded_style_full[b].clone().detach()
                        encoded_style_in[region_idx] = (encoded_style_full[semantics_b, region_idx]).clamp(-1, 1)
                        fake_image, _, _ = self.generate_fake(input_semantics=input_semantics[b].unsqueeze(0),
                                                                 image_downsized=image_lr[b].unsqueeze(0),
                                                                 encoded_style=encoded_style_in.unsqueeze(0))
                        fake_samples.append(fake_image)
                    fake_images.append(torch.cat(fake_samples, -1))
                fake_out = torch.cat(fake_images, 0)
                out = OrderedDict([("input_label", input_semantics),
                             ("image_downsized", image_lr),
                             ("fake_image", fake_out),
                             ("image_full", image_hr)])
                if self.opt.guiding_style_image:
                    out['guiding_image_id'] = data['guiding_image_id']
                    out['guiding_image'] = data['guiding_image']
                    out['guiding_input_label'] = data['guiding_label']
                return out
        elif mode == "inference_reference_interpolation":
            with torch.no_grad():
                # encoded_style_mini, _, _, _ = self.encode_style(downscaled_image=image_downsized,
                #                                            input_semantics=input_semantics, full_image=None,
                #                                            no_noise=True)
                encoded_style_full, _ = self.encode_style(downscaled_image=None,
                                                           input_semantics=input_semantics, full_image=image_hr,
                                                           no_noise=True, encode_full=True)
                region_idx = self.opt.region_idx if self.opt.region_idx else list(range(input_semantics.size(1)))
                fake_images = list()
                for b in range(self.opt.batchSize):
                    fake_samples = list()
                    idx_style_a = (b ) % self.opt.batchSize
                    style_a = encoded_style_full[idx_style_a].clone().detach()
                    idx_style_b = (b + 1) % self.opt.batchSize
                    semantics_b = encoded_style_full[idx_style_b].clone().detach() * self.opt.manipulate_scale
                    for delta_step in np.linspace(0, 1, num=self.opt.n_interpolation):
                        encoded_style_in = style_a
                        encoded_style_in[region_idx] = ((1 - delta_step) * style_a[region_idx] + delta_step * semantics_b[region_idx]).clamp(-1, 1)
                        fake_image, _, _ = self.generate_fake(input_semantics=input_semantics[b].unsqueeze(0),
                                                                 image_downsized=image_lr[b].unsqueeze(0),
                                                                 encoded_style=encoded_style_in.unsqueeze(0))
                        fake_samples.append(fake_image)
                    fake_images.append(torch.cat(fake_samples, -1))
                fake_out = torch.cat(fake_images, 0)
                out = OrderedDict([("input_label", input_semantics),
                             ("image_downsized", image_lr),
                             ("fake_image", fake_out),
                             ("image_full", image_hr)])
                if self.opt.guiding_style_image:
                    out['guiding_image_id'] = data['guiding_image_id']
                    out['guiding_image'] = data['guiding_image']
                    out['guiding_input_label'] = data['guiding_label']
                return out
        else:
            raise ValueError("|mode| is invalid")

    def get_noise(self, shape, delta):
        if self.opt.noise_dist == 'normal':
            noise = torch.randn(shape).clamp(-1, 1) * delta
        elif self.opt.noise_dist == 'uniform':
            noise = torch.rand(shape).clamp(-1, 1) * delta
        else:
            raise ValueError("Invalid noise distribution: {}".format(self.opt.noise_dist))
        if self.use_gpu():
            noise = noise.cuda()
        return noise

    def corrupt_style(self, style, eps=0.05, dist='gaussian'):
        if dist == 'gaussian':
            scale = networks.np.sqrt(eps)
            style_corrupted = torch.randn_like(style) * scale + style
        elif dist == 'uniform':
            scale = networks.np.sqrt(eps)
            style_corrupted = (torch.rand_like(style) * 2 - 1) * scale * 1.4 + style
        delta = torch.nn.functional.mse_loss(style, style_corrupted)
        return style_corrupted

    def create_optimizers(self, opt):
        SR_params = list(self.netSR.parameters())
        SR_params_low_lr = list()
        if self.use_E:
            named_params = list(self.netE.named_parameters())
            for name, param in named_params:
                if "mini" in name:
                    SR_params_low_lr.append(param)
                else:
                    SR_params.append(param)
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            SR_lr, D_lr = opt.lr, opt.lr
        else:
            SR_lr, D_lr = opt.lr / 2, opt.lr * 2
        print("lr G: {}, lr D: {}".format(SR_lr, D_lr))
        optimizer_SR = torch.optim.Adam([
            # We use a lower learning rate for some parameters
            {"params": SR_params},
            {"params": SR_params_low_lr, "lr": SR_lr / 4}
        ], lr=SR_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_SR, optimizer_D

    def save(self, epoch):
        util.save_network(self.netSR, 'SR', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.use_E:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        self.netSR = networks.define_SR(opt)
        self.netD = networks.define_D(opt) if opt.isTrain else None
        self.netE = networks.define_E(opt) if self.use_E else None

        self.load_weights()
        return self.netSR, self.netD, self.netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data
    def compute_generator_loss(self, input_semantics, image_full, image_downsized, guiding_image, guiding_label):
        SR_losses = {}
        style_image = guiding_image if self.opt.guiding_style_image else image_full
        fake_image, encoder_activations_real, encoded_style = self.generate_fake(
            input_semantics=input_semantics, image_downsized=image_downsized, full_image=style_image, guiding_image=guiding_image, guiding_label=guiding_label)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, image_full)
        SR_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            SR_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            SR_losses['VGG'] = self.criterionVGG(fake_image, image_full) \
                * self.opt.lambda_vgg

        return SR_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, image_full, image_downsized, guiding_image, guiding_label):
        gpu_info("D loss start", self.opt)
        D_losses = {}
        with torch.no_grad():
            fake_image, _, _ = self.generate_fake(input_semantics=input_semantics, image_downsized=image_downsized, full_image=image_full, guiding_image=guiding_image, guiding_label=guiding_label)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        gpu_info("D loss after generate", self.opt)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, image_full)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_Real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)
        gpu_info("D loss end", self.opt)
        return D_losses

    def generate_fake(self, input_semantics, image_downsized, encoded_style=None, full_image=None, no_noise=False, guiding_image=None, guiding_label=None):
        gpu_info("Generate fake start", self.opt)
        encoder_activations = None
        if encoded_style is None and "style" in self.opt.netE:
            encoded_style, encoder_activations = self.encode_style(downscaled_image=image_downsized, input_semantics=input_semantics, full_image=full_image, no_noise=no_noise, guiding_image=guiding_image, guiding_label=guiding_label, encode_full=self.opt.full_style_image)
        gpu_info("Generate fake before SR", self.opt)

        fake_image = self.netSR(image_downsized, seg=input_semantics, z=encoded_style)
        gpu_info("Generate fake after SR", self.opt)

        if self.mp == 1:
            fake_image = fake_image.cuda(0)
            encoded_style = encoded_style.cuda(0)

        return fake_image, encoder_activations, encoded_style

    def get_encoder_inputs(self, downscaled_image=None, input_semantics=None, full_image=None, encode_full=False, guiding_image=None, guiding_label=None):
        """
        Chooses the correct inputs for the style encoder.
        Args:
            downscaled_image: LR input image
            input_semantics:  HR semantics for LR input image
            full_image: Ground truth of LR input image
            encode_full: flag whether to use an HR style image.
                Only used for the independent model.
            guiding_image: HR style image
            guiding_label: HR semantics for style image
        Returns: style image, semantics for style image

        """
        # might be none: TODO: debug
        # Default choices
        style_semantics = input_semantics
        style_image = downscaled_image

        if self.model_variant == "guided":
            mode = "full"
            # We choose the high-resolution input based
            # on the "guiding_style_image" flag
            if self.opt.guiding_style_image:
                style_semantics = guiding_label
                style_image = guiding_image
            else:
                # Semantics is already assigned correctly
                style_image = full_image
        elif self.model_variant == "independent":
            # encode_full can be used at demo time to
            # enforce a HR style image
            # During training, we take a high-resolution style image
            # in 50% of the iterations
           if encode_full or (self.training and random.random() < 0.5):
               mode = "full"
               # We choose the high-resolution input based
               # on the "guiding_style_image" flag
               self.last_encoded_style_is_full = True
               if self.opt.guiding_style_image:
                   style_semantics = guiding_label
                   style_image = guiding_image
               else:
                   # Semantics is already assigned correctly
                   style_image = full_image
           else:
               mode = "mini"
               self.last_encoded_style_is_full = False
        else:
            raise NotImplementedError()
        return style_image, style_semantics, mode

    def encode_style(self, downscaled_image=None, input_semantics=None, full_image=None, encode_full=False, no_noise=None, guiding_image=None, guiding_label=None):
        style_image, style_semantics, mode = self.get_encoder_inputs(downscaled_image=downscaled_image, input_semantics=input_semantics, full_image=full_image, encode_full=encode_full, guiding_image=guiding_image, guiding_label=guiding_label)

        if self.model_variant == "guided":
            encoded_style, activations = self.netE(style_image, style_semantics, mode=mode, no_noise=no_noise)
        elif self.model_variant == "independent":
            # In the combined mode, we use the guiding label in case of the full image and the input_semantics in case of the downsized image
            if not no_noise:
                # train with and without noise
                no_noise = random.random() < 0.5
                self.last_encoded_style_is_noisy = not no_noise
            encoded_style, activations = self.netE(style_image, style_semantics,
                                                   mode=mode,
                                                   no_noise=no_noise)
        else:
            raise NotImplementedError()
        return encoded_style, activations

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
