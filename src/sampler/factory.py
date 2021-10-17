from src.factory import Factory
from src.log import myLogger
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler


class SamplerFactory(Factory[Sampler]):
    def __init__(self, sampler_type: str, logger: myLogger, **kwargs):
        super().__init__(
            sampler_type=sampler_type, logger=logger,
        )

    def _create(self, dataset: Dataset) -> Sampler:
        if self.sampler_type == "sequential":
            sampler = SequentialSampler(data_source=dataset)
        elif self.sampler_type == "random":
            sampler = RandomSampler(data_source=dataset)
        else:
            raise NotImplementedError
        return sampler
