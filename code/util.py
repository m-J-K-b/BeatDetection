import time


def time_it(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"executing function '{func.__name__}' took: {end - start}s")
        return result

    return new_func
