from data.preprocessor import Preprocessor

from deepsee_models.networks.sync_batchnorm import DataParallelWithCallback

from deepsee_models.sr_model import SRModel


class BaseManager:
    def __init__(self, opt, create_model=True):
        self.opt = opt
        self.preprocessor = Preprocessor(opt)
        if create_model:
            self.create_model(opt)

    def create_model(self, opt):
        self.sr_model = SRModel(opt)
        if len(
                opt.gpu_ids) > 0 and opt.model_parallel_mode == 0:  # not using model parallelism
            self.sr_model = DataParallelWithCallback(self.sr_model,
                                                     device_ids=opt.gpu_ids)
            self.sr_model_on_one_gpu = self.sr_model.module
        else:
            self.sr_model_on_one_gpu = self.sr_model

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def preprocess(self, data, from_dataloader=False):
        data = self.preprocess_datatypes(data)
        data = self.preprocess_gpu(data)
        if from_dataloader:
            data = self.preprocess_from_dataloader(data)
        return data

    def preprocess_datatypes(self, data):
        # Change data types
        for k in data:
            if 'label' in k or 'semantics' in k:
                data[k] = data[k].long()
        return data

    def preprocess_gpu(self, data):
        # moves to GPU, if needed
        if self.use_gpu():
            for k in data:
                if "label" in k or "semantics" in k or "image" in k:
                    data[k] = data[k].cuda()
        return data

    def preprocess_from_dataloader(self, data):
        # create one-hot label map
        input_semantics = self.preprocessor.preprocess_label(data['label'])

        image_downsized = self.preprocessor.downsample_image(data['image'])
        data_preprocessed = {
            "input_semantics": input_semantics,
            "image_lr": image_downsized,
            "image_hr": data["image"]
        }

        if self.opt.guiding_style_image:
            data_preprocessed["guiding_image"] = data['guiding_image']
            data_preprocessed[
                "guiding_label"] = self.preprocessor.preprocess_label(
                data['guiding_label'].long())
        return data_preprocessed
