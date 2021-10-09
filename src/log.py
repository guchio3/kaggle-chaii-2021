import os
from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger
from typing import Optional

import requests


class myLogger:
    def __init__(self, log_filename: str) -> None:
        self.logger = getLogger(__name__)
        log_dir_name = "/".join(log_filename.split("/")[:-1])
        if not os.path.exists(log_dir_name):
            os.makedirs(log_dir_name)
        self._logInit(log_filename)

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
