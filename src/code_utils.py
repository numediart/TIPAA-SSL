import os, psutil

print_memory_usage = lambda stage: print(
    stage + ": " + str(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
)

import sys, traceback


def internal_error():
    """This can be called in a try-except to print info about the exception  (type, value, traceback)"""
    etype, value, tb = sys.exc_info()

    content = {
        'type': str(etype),
        'value': str(value),
        'traceback': str(traceback.format_tb(tb)),
    }
    return content
