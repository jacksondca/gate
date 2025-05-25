import time

from functools import wraps


def print_time(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f"`{func.__name__}` time cost: {time.time() - start}")
    
    return wrapper