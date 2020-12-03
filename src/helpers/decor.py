from functools import wraps
import time
import logging

def function_time(func):
    """Decorator for logging node execution time and event contents
        Args:
            func: Function to be executed.
        Returns:
            Decorator for logging the running time.
    """

    @wraps(func)
    def with_time(*args, **kwargs):
        log = logging.getLogger(__name__)

        logging.info(f"Starting function {func.__name__}")

        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()
        elapsed = t_end - t_start
        log.info("Running %r took %.2f seconds", func.__name__, elapsed)
        return result

    return with_time
