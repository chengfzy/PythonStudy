"""
Some code for B-Spline curve

S(x) = Sum_{j=0}^{n-1} c[j] * B[j,k;t](x)

B[i,0](x) = 1 if t[i] <= x <= t[i+1], otherwise 0
B[i,k](x) = (x - t[i]) / (t[i+k] - t[i]) * B[i, k-1](x) + (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * B[i+1, k-1](x)

Ref:
[1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
"""

from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np


def basic(x, i, k, t):
    """Basic elements in B-Spline"""
    if k == 0:
        return 1.0 if t[i] <= x < t[i + 1] else 0.0

    if t[i + k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i]) / (t[i + k] - t[i]) * basic(x, i, k - 1, t)

    if t[i + k + 1] == t[i + 1]:
        c2 = 0.0
    else:
        c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * basic(x, i + 1, k - 1, t)

    return c1 + c2


def bspline(x, t, c, k):
    """B-Spline function"""
    n = len(t) - k - 1
    assert (n >= k + 1) and (len(c) >= n)
    return sum(c[i] * basic(x, i, k, t) for i in range(n))


def main():
    k = 2
    t = [0, 1, 2, 3, 4, 5, 6]
    c = [-1, 2, 0, -1]
    sp1 = BSpline(t, c, k, True)  # if extrapolate is False, they are the same
    print(f'BSpline(2.5) = {sp1(2.5)}, bspline(2.5) = {bspline(2.5, t, c, k)}')
    xx = np.linspace(1.5, 4.5, 50)
    xx_fine = np.linspace(1.5, 4.5, 500)

    # figure, B-Spline basic function
    fig = plt.figure('B-Spline Basic Function')
    ax = fig.add_subplot(111)
    ax.plot(xx_fine, [basic(x, 2, 0, t) for x in xx_fine], 'k-', label='degree = 0')
    ax.plot(xx_fine, [basic(x, 2, 1, t) for x in xx_fine], 'b-', label='degree = 1')
    ax.plot(xx_fine, [basic(x, 2, 2, t) for x in xx_fine], 'r-', label='degree = 2')
    ax.grid(True)
    ax.set_title('B-Spline Basic Function B[2, k]')
    ax.legend(loc='best')
    plt.show(block=False)

    # figure: B-Spline
    fig = plt.figure('B-Spline')
    ax = fig.add_subplot(111)
    ax.plot(xx, [bspline(x, t, c, k) for x in xx], 'r-', label='naive')
    ax.plot(xx, sp1(xx), 'b.-', label='BSpline')
    ax.grid(True)
    ax.set_title('B-Spline')
    ax.legend(loc='best')
    plt.show(block=True)


if __name__ == '__main__':
    main()
