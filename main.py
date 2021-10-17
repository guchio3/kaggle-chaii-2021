import argparse
import os
import random

import numpy as np

import torch
from src.pipeline.factory import PipelineFactory


def parse_args():
    """
    Policy
    ------------
    * experiment id must be required
    """
    parser = argparse.ArgumentParser(
        prog="XXX.py",
        usage="ex) python main.py -e e001 -p train",
        description="short explanation of args",
        add_help=True,
    )
    parser.add_argument(
        "-e", "--exp_id", help="experiment setting", type=str, required=True
    )
    parser.add_argument(
        "-d",
        "--device",
        help="cpu or cuda, the device for running the model",
        type=str,
        required=False,
        default="cuda",
    )
    parser.add_argument(
        "-p",
        "--pipeline_type",
        help="the pipeline type, choose from (train|pred)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--debug",
        help="whether or not to use debug mode",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    return args


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