"""
Solve least square function
Ref: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#least-squares-minimization-least-squares
"""

import numpy as np
from scipy.optimize import least_squares
import util


def model(x, u):
    return x[0] * (u**2 + x[1] * u) / (u**2 + u * x[2] + x[3])


def cost_fun(x, u, y):
    return model(x, u) - y


def jac(x, u, y):
    J = np.empty((u.size, x.size))
    den = u**2 + u * x[2] + x[3]
    num = u**2 + u * x[1]
    J[:, 0] = num / den
    J[:, 1] = u * x[0] / den
    J[:, 2] = -u * x[0] * num / den**2
    J[:, 3] = -x[0] * num / den**2
    return J


def solve_least_square():
    print(util.Section('Solve Least Square'))
    u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2])
    y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
    x0 = np.array([2.5, 3.9, 4.15, 3.9])
    res = least_squares(cost_fun, x0, jac=jac, bounds=(0, 100), args=(u, y), verbose=2)
    print(f'Result: {res.x}')


if __name__ == '__main__':
    solve_least_square()
