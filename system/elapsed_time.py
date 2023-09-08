"""
Get the elapsed time
"""
import time
from datetime import timedelta

if __name__ == '__main__':
    # start time
    t0 = time.time()

    # do some work
    time.sleep(5)

    # end time
    s0 = time.time()
    x0 = s0 - t0
    print(f'used time = {x0:.3f}s')
    print(f'used time = {timedelta(seconds=s0-t0)}')
