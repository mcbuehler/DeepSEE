import torch
import torch.nn.functional as F


class Preprocessor:
    def __init__(self, opt):
        self.opt = opt

        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def downsample_image(self, hr_image, shape=None):
        """
        Downsamples a HR image to a small square, as indicated by self.opt.start_size.
        Args:
            hr_image: Tensor with HR image in [-1, 1]
            shape: (height, width) of output image

        Returns: LR image in [-1, 1]
        """
        if shape is None:
            shape = (self.opt.start_size, self.opt.start_size)

        image_lr = F.interpolate(hr_image,shape,
                      mode=self.opt.downsampling_method)
        # Avoid overshooting
        image_lr = image_lr.clamp(min=-1, max=1)
        return image_lr

    def preprocess_label(self, label_map):
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        return input_semantics

