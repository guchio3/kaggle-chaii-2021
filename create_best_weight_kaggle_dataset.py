import argparse
import json
import os

from src.log import myLogger
from src.repository.data_repository import DataRepository


def parse_args():
    """
    Policy
    ------------
    * experiment id must be required
    """
    parser = argparse.ArgumentParser(
        prog="XXX.py",
        usage="ex) python main.py -e e001",
        description="short explanation of args",
        add_help=True,
    )
    parser.add_argument(
        "-e",
        "--exp_ids",
        help="experiment setting",
        type=str,
        nargs="+",
        required=True,
    )
    args = parser.parse_args()
    return args


def create_dataset(exp_id: str):
    os.system("cp -r .kaggle ~")
    os.system("chmod 600 ~/.kaggle/kaggle.json")
    # download
    logger = myLogger(
        log_filename="./logs/create_best_weight_kaggle_dataset.log",
        exp_id="",
        wdb_prj_id="kaggle-chaii-2021",
        exp_config={},
        use_wdb=False,
    )
    dr = DataRepository(logger=logger)
    dr.download_best_model_state_dicts(exp_id=exp_id)

    # create kaggle dataset
    exp_checkpoint_root_path = f"./data/checkpoint/{exp_id}"
    os.system(f"mkdir -p {exp_checkpoint_root_path}")
    os.system(f"kaggle datasets init -p {exp_checkpoint_root_path}")
    with open(f"{exp_checkpoint_root_path}/dataset-metadata.json", "r") as fin:
        dataset_metadata = json.load(fin)
    dataset_metadata["title"] = f"{exp_id}_best_weights"
    dataset_metadata["id"] = f"guchio3/{exp_id}-best-weights"
    with open(f"{exp_checkpoint_root_path}/dataset-metadata.json", "w") as fout:
        json.dump(dataset_metadata, fout)
    os.system(f"kaggle datasets create -p {exp_checkpoint_root_path} -r tar")

    # remove
    dr.clean_exp_checkpoint(exp_id=exp_id, delete_from_gcs=False)


def main():
    args = parse_args()
    for exp_id in args.exp_ids:
        create_dataset(exp_id=exp_id)


if __name__ == "__main__":
    main()
