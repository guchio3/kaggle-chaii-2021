import functools
import traceback
from typing import Callable


def class_error_line_notification(add_traceback: bool) -> Callable:
    """
    can only used for functions which return None.
    """

    def _dec_try_and_notify_errors(func: Callable) -> Callable:
        # wraps func enable to hold the func name
        @functools.wraps(func)
        def _try_and_notify_errors(s, *args, **kwargs) -> None:
            # run the func
            try:
                func(s, *args, **kwargs)
            except Exception as e:
                message = str(e)
                if add_traceback:
                    message = f"{e}\n{traceback.format_exc()}"
                s.logger.error(message)
                s.logger.send_line_notification(message=message)

        return _try_and_notify_errors

    return _dec_try_and_notify_errors
