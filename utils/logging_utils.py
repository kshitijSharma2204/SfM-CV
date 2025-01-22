# utils/logging_utils.py

import logging
import time
from functools import wraps

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
    )
    logger = logging.getLogger()
    return logger

def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
