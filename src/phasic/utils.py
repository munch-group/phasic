
from functools import wraps, partial
import time

def hand_off(target_func):
    """Decorator that forwards all parameters to a another function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return target_func(*args, **kwargs)
        return wrapper
    return decorator

