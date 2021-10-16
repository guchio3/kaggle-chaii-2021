import random

import os

import numpy as np

import torch
from src.args import parse_args
from src.pipeline.factory import PipelineFactory


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(1213)


def main():
    args = parse_args()
    factory = PipelineFactory()
    pipeline = factory.create(
        pipeline_type=args.pipeline_type,
        exp_id=args.exp_id,
        device=args.device,
        debug=args.debug,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
