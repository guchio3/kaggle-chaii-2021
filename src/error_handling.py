import functools
import traceback
from typing import Callable, TypeVar

T = TypeVar("T")


def class_error_line_notification(add_traceback: bool, return_value: T) -> Callable:
    """
    can only used for functions which return None.
    """

    def _dec_try_and_notify_errors(func: Callable) -> Callable:
        # wraps func enable to hold the func name
        @functools.wraps(func)
        def _try_and_notify_errors(s, *args, **kwargs) -> T:
            # run the func
            try:
                res: T = func(s, *args, **kwargs)
            except Exception as e:
                res = return_value
                message = str(e)
                if add_traceback:
                    message = f"{e}\n{traceback.format_exc()}"
                s.logger.error(message)
                s.logger.send_line_notification(message=message)
            return res

        return _try_and_notify_errors

    return _dec_try_and_notify_errors
