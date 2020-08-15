import time


def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            time.sleep(.1) 
            it = iter(iterable)


