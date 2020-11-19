from .base_manager import BaseManager


class DemoManager(BaseManager):
    """

    """
    def __init__(self, opt):
        super().__init__(opt)
        self.sr_model = self.sr_model.eval()

    def compute_style_from_hr(self, inputs_hr):
        print("Encoding style from {} HR images...".format(len(inputs_hr)))
        inputs_hr = [super(DemoManager, self).preprocess_input(inputs_hr[i]) for i in range(len(inputs_hr))]
        all_encoded_styles = list()
        # Compute style matrix for all inputs
        for input_hr in inputs_hr:
            data_preprocessed = {"image_guiding": input_hr["image_hr"], "label_guiding": self.preprocessor.preprocess_label(input_hr["semantics"])}
            all_encoded_styles.append(self.sr_model.forward(data_preprocessed, "encode_only"))

        # We take the full style matrix of the first entry...
        encoded_style = all_encoded_styles[0]
        # ...and replace some rows with styles from other HR images.
        for input_i in range(1, len(inputs_hr)):
            regions = inputs_hr[input_i]["regions"]
            for region_index in regions:
                encoded_style[:, region_index] = all_encoded_styles[input_i][:, region_index].detach()
        encoded_style = encoded_style.clone()
        return encoded_style

    def compute_style_from_lr(self, data):
        print("Encoding style from LR image...")
        data = super().preprocess(data, from_dataloader=False)
        data_preprocesed = {"image_lr": data["image_lr"], "input_semantics": self.preprocessor.preprocess_label(data["input_semantics"])}
        encoded_style = self.sr_model.forward(data_preprocesed, "encode_only")
        return encoded_style

    def run(self, data):
        assert "image_lr" in data.keys()
        assert "semantics" in data.keys()
        assert "encoded_style" in data.keys()
        data = super().preprocess(data, from_dataloader=False)

        data_preprocessed = {
            "image_lr": data["image_lr"],
            "input_semantics": self.preprocessor.preprocess_label(data["semantics"]),
            "encoded_style": data["encoded_style"]
        }
        return self.sr_model.forward(data_preprocessed, "demo")


