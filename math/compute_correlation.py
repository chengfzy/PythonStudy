"""
Compute and Visualize the Time & Frequency Correlation

Ref:
    [1] https://towardsdatascience.com/computing-cross-correlation-between-geophysical-time-series-488642be7bf0
    [2] https://github.com/earthinversion/geophysical-cross-correlation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal


def cross_correlation(x: pd.Series, y: pd.Series, lag=0) -> float:
    """
    Lag-N cross correlation. Shift data filled with NaNs

    Args:
        x (pandas.Series): data x
        y (pandas.Series): data y
        lag (int, optional): lag. Defaults to 0.
    """
    return x.corr(y.shift(lag))


def cross_correlation_using_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate cross correlation using FFT

    Args:
        x (np.ndarray): data x
        y (np.ndarray): data y

    Returns:
        np.ndarray: the crossed correlation data
    """
    f1 = np.fft.fft(x)
    f2 = np.fft.fft(np.flipud(y))
    cc = np.real(np.fft.ifft(f1 * f2))
    return np.fft.fftshift(cc)


def compute_shift(x: np.ndarray, y: np.ndarray):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift


def test01():
    """
    Compute time domain cross-correlation between two time series
    """
    # delta function
    length = 100
    amp1, amp2 = 1, 1
    x = np.arange(0, length)
    t0 = 10
    timeshift = 30
    t1 = t0 + timeshift
    series0 = signal.unit_impulse(length, idx=t0)
    series1 = signal.unit_impulse(length, idx=t1)

    # low pass filter to smoothen the edges(just to make the signal look pretty)
    b, a = signal.butter(4, 0.2)
    series0 = signal.lfilter(b, a, series0)
    series1 = signal.lfilter(b, a, series1)

    # plot series
    # plt.style.use('seaborn')
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    ax[0].plot(x, series0, c='b', lw=0.5)
    ax[0].axvline(x=t0, c='b', lw=0.5, ls='--', label=f'x={t0}')
    ax[0].plot(x, series1 + 0.1, c='r', lw=0.5)
    ax[0].axvline(x=t1, c='r', lw=0.5, ls='--', label=f'x={t1}')
    ax[0].set_yticks([0, 0.1])
    ax[0].legend()
    ax[0].set_yticklabels(['Series0', 'Series1'], fontsize=8)

    # compute correlation
    d1, d2 = pd.Series(series0), pd.Series(series1)
    lags = np.arange(-50, 50, 1)
    rs = np.nan_to_num([cross_correlation(d1, d2, lag) for lag in lags])
    maxrs, minrs = np.max(rs), np.min(rs)
    if np.abs(maxrs) >= np.abs(minrs):
        corrval = maxrs
    else:
        corrval = minrs
    # compute shift
    shift = compute_shift(series0, series1)
    print(f'shift = {shift}')

    # plot correlation
    ax[1].plot(lags, rs, 'k', label=f'Xcorr, maxcorr = {corrval:.2f}', lw=0.5)
    ax[1].axvline(x=lags[np.argmax(rs)], c='r', lw=0.5, ls='--', label='Correlation')
    ax[1].legend(fontsize=6)
    plt.subplots_adjust(hspace=0.25, wspace=0.1)


if __name__ == '__main__':
    test01()

    plt.show(block=True)
