import functools
import time
from typing import Callable

from src.log import myLogger


def dec_timer(unit: str = "s") -> Callable:
    def _dec_timer(func: Callable) -> Callable:
        # wraps func enable to hold the func name
        @functools.wraps(func)
        def _timer(*args, **kwargs):
            if "logger" not in kwargs:
                raise Exception("You must include logger: myLogger in the args.")
            logger: myLogger = kwargs["logger"]

            t0 = time.time()
            start_str = f"[{func.__name__}] start"
            logger.info(start_str)

            # run the func
            res = func(logger=logger, *args, **kwargs)

            if unit == "s":
                duration = time.time() - t0
            elif unit == "m":
                duration = (time.time() - t0) / 60
            elif unit == "h":
                duration = (time.time() - t0) / 3600
            else:
                raise NotImplementedError()
            end_str = f"[{func.__name__}] done in {duration:.1f} {unit}"
            logger.info(end_str)
            return res

        return _timer

    return _dec_timer


def class_dec_timer(unit: str = "s") -> Callable:
    def _dec_timer(func: Callable) -> Callable:
        # wraps func enable to hold the func name
        @functools.wraps(func)
        def _timer(s, *args, **kwargs):
            t0 = time.time()
            start_str = f"[{func.__name__}] start"
            s.logger.info(start_str)

            # run the func
            res = func(s, *args, **kwargs)

            if unit == "s":
                duration = time.time() - t0
            elif unit == "m":
                duration = (time.time() - t0) / 60
            elif unit == "h":
                duration = (time.time() - t0) / 3600
            else:
                raise NotImplementedError()
            end_str = f"[{func.__name__}] done in {duration:.1f} {unit}"
            s.logger.info(end_str)
            return res

        return _timer

    return _dec_timer
