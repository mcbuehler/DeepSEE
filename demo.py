import io
import json
import os
from collections import OrderedDict

import ipywidgets as widgets
import numpy as np
import torch
from PIL import Image
from ipywidgets import interact

from data.base_dataset import get_transform, get_params
from managers.demo_manager import DemoManager
from options.configurations import get_opt_config
from util.util import mkdirs, get_celebA_region_name, save_image, \
    save_style_matrix, ObjectDict, Colorize, tensor2im
from util.visualizer import Visualizer


class Demo():
    def __init__(self, opt):
        self.opt = opt
        self.manager = DemoManager(opt)

        self.save_dir = os.path.join(opt.results_dir, opt.name, "demo", opt.dataset)

        mkdirs(self.save_dir)
        self.region_str = "_".join(
            map(lambda i: get_celebA_region_name(i, nospace=True),
                opt.region_idx)) if opt.region_idx else "all"

    def load_image(self, path, params, preprocess_mode=None):
        image = Image.open(path)
        image = image.convert('RGB')
        transform_image = get_transform(self.opt, params, preprocess_mode=preprocess_mode)
        image_tensor = transform_image(image)
        return image_tensor.unsqueeze(0)

    def load_style(self, path):
        style = np.loadtxt(path, delimiter=",")
        tensor = torch.from_numpy(style)
        return tensor.unsqueeze(0).float()

    def compute_style_from_hr(self, inputs_hr):
        ...  # TODO

    def compute_style_from_lr(self, inputs_hr):
        ...  # TODO

    def load_label(self, path, params):
        label = Image.open(path)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST,
                                        normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[
            label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        return label_tensor.unsqueeze(0)

    def get_id_from_path(self, path):
        return path.split('/')[-1][:-4]

    def save_result(self, results, **kwargs):
        visuals_np = Visualizer.convert_visuals_to_numpy(results, batchSize=1,
                                                         label_nc=self.opt.label_nc)
        # We only run the demo with batch Size 1, so let's remove the first dimension.
        visuals_np = OrderedDict([(k, v[0]) for k, v in visuals_np.items()])

        filename = self._get_filename(kwargs)
        save_path = os.path.join(self.save_dir, filename)
        save_image(visuals_np["fake_image"], save_path, create_dir=True)

        save_style_matrix(results["encoded_style"][0], "{}.csv".format(save_path[:-4]))
        return self.save_dir

    def _get_filename(self, kwargs):
        lr_input = os.path.basename(kwargs["path_image_lr"])[:-4]
        name = "{}_lr_{}".format(kwargs["name"], lr_input)
        if kwargs.get("path_encoded_style", ''):
            filename = "{}_encoded_style_{}.png".format(name, os.path.basename(kwargs["path_encoded_style"])[:-4])
        elif len(kwargs.get('inputs_hr', [])) > 0:
            hr_filenames = []
            for i in range(len(kwargs.get('inputs_hr'))):
                file_basename = self.get_id_from_path(kwargs["inputs_hr"][i]["path_image_hr"])
                regions = kwargs["inputs_hr"][i]["regions"]
                regions = "-".join(map(str, regions)) if regions != "all" else "all"
                hr_filenames.append("{}-{}".format(file_basename, regions))
            filename = "{}_hr_{}.png".format(name, "_".join(hr_filenames))
        else:
            filename = "{}_independent.png".format(name)
        return filename

    def run(self, **kwargs):
        params = get_params(self.opt, (self.opt.crop_size, self.opt.crop_size))
        image_lr = self.load_image(kwargs['path_image_lr'], params, preprocess_mode="none")
        semantics = self.load_label(kwargs['path_semantics'], params)

        if kwargs.get('path_encoded_style', '') != '':
            # Load from CSV file
            encoded_style = self.load_style(kwargs.get('path_encoded_style'))
        elif kwargs.get('encoded_style', '') != '':
            encoded_style = kwargs['encoded_style']
        elif len(kwargs.get('inputs_hr', [])) > 0:
            # Compute style from one or more HR images
            inputs_hr = kwargs.get('inputs_hr')
            inputs_hr = [
                {'image_hr': self.load_image(input_hr['path_image_hr'], params),
                 'semantics': self.load_label(input_hr['path_semantics'], params),
                 'regions': input_hr['regions']
                 } for input_hr in inputs_hr
            ]
            encoded_style = self.manager.compute_style_from_hr(inputs_hr)
            print("Style computed.")
        else:
            # Compute style from the LR input
            assert self.opt.netE == "combinedstyle", "Only the independent model can compute the style from a LR image."
            inputs = {"image_lr": image_lr, "input_semantics": semantics}
            encoded_style = self.manager.compute_style_from_lr(inputs)
            print("Style computed.")

        # TODO: add noise and modifications of style matrix
        input_dict = {
            'image_lr': image_lr,
            'semantics': semantics,
            'encoded_style': encoded_style
        }
        print("Upscaling...")
        result = self.manager.run(input_dict)
        save_path = self.save_result(result, **kwargs)
        result["save_path"] = save_path
        print("Done.")
        return result


def get_demo_options(name, path="options/demo_options.json"):
    with open(path, "r") as f:
        opt = json.load(f)

    opt = ObjectDict(opt)
    # Update config based on opt.name
    opt.name = name
    opt = get_opt_config(opt, opt.name)
    return opt


def tensor2label(label):
    label = label.max(0, keepdim=True)[1]
    label = Colorize(19)(label)
    label = np.transpose(label.numpy(), (1, 2, 0))
    label = label.astype(np.uint8)
    return label


def process_result(result, key):
    if key in ["fake_image"]:
        image = tensor2im(result[key][0])
        return image
    if key in ["image_lr"]:
        image = tensor2im(result[key][0])
        hr_size = result["fake_image"].shape[-1], result["fake_image"].shape[-2]
        return np.array(Image.fromarray(image).resize(hr_size, Image.NEAREST))
    if key in ["input_semantics"]:
        return tensor2label(result[key][0])
    if key in ["encoded_style"]:
        result["encoded_style"][0] = result["encoded_style"][0] / result["encoded_style"][0].abs().max()
        return ((result["encoded_style"][0].detach().cpu().numpy() + 1) * 127.5).astype(np.uint8)


def image_to_byte_array(image):
    img_bytes = io.BytesIO()
    image = Image.fromarray(image)
    image.save(img_bytes, format="png")
    return img_bytes.getvalue()


def interact_f(x, result, size=(512, 512)):
    image = process_result(result, x)
    image = image_to_byte_array(image)
    return widgets.Image(
        value=image,
        format='png',
        width=size[0],
        height=size[1],
    )


def display_result(result, size=(512, 512)):
    w = widgets.Dropdown(
        options=sorted(["image_lr", "fake_image", "input_semantics", "encoded_style"]),
        value='fake_image',
        description='Visualize:',
        disabled=False, )
    interact(lambda x: interact_f(x, result, size), x=w)
