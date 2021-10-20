import os
from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger
from typing import Any, Dict, Optional

import requests

import wandb


class myLogger:
    def __init__(
        self,
        log_filename: str,
        exp_id: str,
        wdb_prj_id: Optional[str],
        exp_config: Dict[str, Any],
        use_wdb: bool,
    ) -> None:
        self.logger = getLogger(__name__)
        log_dir_name = "/".join(log_filename.split("/")[:-1])
        if not os.path.exists(log_dir_name):
            os.makedirs(log_dir_name)
        self._logInit(log_filename)

        self.use_wdb = use_wdb
        if self.use_wdb and wdb_prj_id:
            self._wandb_init(wdb_prj_id=wdb_prj_id, exp_id=exp_id)
        else:
            self.info("skip wandb init")

    def _wandb_init(self, wdb_prj_id: str, exp_id: str) -> None:
        wandb.init(project=wdb_prj_id, entity="guchio3")
        wandb.run.name = exp_id
        wandb.run.save()

        # define our custom x axis metric
        wandb.define_metric("epoch")
        # define which metrics will be plotted against it
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("valid/*", step_metric="epoch")

    def info(self, message: str) -> None:
        self.logger.info(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def warn(self, message: str) -> None:
        self.logger.warn(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def send_line_notification(self, message: str) -> None:
        self.logger.info(message)
        line_token = ""
        endpoint = "https://notify-api.line.me/api/notify"
        message = "\n{}".format(message)
        payload = {"message": message}
        headers = {"Authorization": "Bearer {}".format(line_token)}
        requests.post(endpoint, data=payload, headers=headers)

    def _logInit(self, log_filename: Optional[str]) -> None:
        log_fmt = Formatter(
            "%(asctime)s %(name)s \
                %(lineno)d [%(levelname)s] [%(funcName)s] %(message)s "
        )
        handler = StreamHandler()
        handler.setLevel("INFO")
        handler.setFormatter(log_fmt)
        self.logger.addHandler(handler)

        if log_filename:
            handler = FileHandler(log_filename, "a")
            handler.setLevel(DEBUG)
            handler.setFormatter(log_fmt)
            self.logger.setLevel(DEBUG)
            self.logger.addHandler(handler)

    def wdb_log(self, log_dict: Dict[str, Any]) -> None:
        if self.use_wdb:
            self.warn("pass wdb_log because debug mode.")
            return
        wandb.log(log_dict)

    def wdb_sum(self, sum_dict: Dict[str, Any]) -> None:
        if self.use_wdb:
            self.warn("pass wdb_sum because debug mode.")
            return
        for key, value in sum_dict.items():
            wandb.run.summery[key] = value
