from torch.nn import Conv1d

from src.model.model import Model


class ChaiiXLMRBModel1(Model):
    def __init__(self, pretrained_model_name_or_path: str) -> None:
        super().__init__(pretrained_model_name_or_path)
        self.classifier_conv_start = Conv1d(self.model.pooler.dense.out_features, 1, 1)
        self.classifier_conv_end = Conv1d(self.model.pooler.dense.out_features, 1, 1)
        self.add_module("conv_output_start", self.classifier_conv_start)
        self.add_module("conv_output_end", self.classifier_conv_end)

    def forward(self, ) -> :
