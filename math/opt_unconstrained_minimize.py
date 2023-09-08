"""
Solve the Rosenbrock function of N variables

Ref: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize
"""

import numpy as np
from scipy.optimize import minimize
import util


def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rosen_jac(x):
    """
    jacobian of rosen function, I don't understand how to calculate it like this
    """
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    jac = np.zeros_like(x)
    jac[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
    jac[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    jac[-1] = 200 * (x[-1] - x[-2] ** 2)
    return jac


def solve_minimize():
    """simple way"""
    print(util.Section('Solve by minimize'))
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='powell', options={'xtol': 1e-8, 'disp': True})
    print(f'Result: {res.x}')


def solve_minimize_jac():
    """solve with jacobian function"""
    print(util.Section('Solve by minimize with Jacobians'))
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, method='BFGS', jac=rosen_jac, options={'disp': True})
    print(f'Result: {res.x}')


if __name__ == '__main__':
    solve_minimize()
    solve_minimize_jac()
