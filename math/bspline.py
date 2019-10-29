"""
Some operation for B-Spline: construction, evaluation, derivation, etc.

Ref:
    [1] https://github.com/orbingol/NURBS-Python_Examples/blob/master/visualization/mpl_curve2d_tangents.py
"""

import util
from geomdl import BSpline
from geomdl import utilities

import numpy as np
import matplotlib.pyplot as plt


class MyBSpline:
    """
    B-Spline created by me
    """

    def __init__(self, degree: int, control_points: np.ndarray):
        self.degree = degree
        self.ctrl_points = control_points
        self._generate_knots()

    def eval(self, u: float):
        """Evaluate B-Splint at u"""
        if 0 > u or u > 1.0:
            raise ValueError(f'input u = {u} should between 0 and 1')

        k = self._find_span_linear(u)
        p = self.degree
        result = self.ctrl_points[k - p: k + 1, :].copy()
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                dev = self.knots[j + k - r + 1] - self.knots[j + k - p]
                alpha = 0.0 if dev == 0 else (u - self.knots[j + k - p]) / dev
                result[j] = (1.0 - alpha) * result[j - 1] + alpha * result[j]
        return result[-1]

    def _generate_knots(self, spline_type='clamp'):
        """Generate uniform knots vector"""
        if 'clamp' == spline_type:
            # m = n + p + 1, and first and last knot must be of multiplicity of p + 1
            p = self.degree
            m = len(self.ctrl_points) + p + 1
            self.knots = np.zeros(m)
            self.knots[p:-p] = np.linspace(0, 1.0, m - 2 * p)
            self.knots[-p:] = 1.0
        else:
            raise NotImplementedError('only support "clamp" BSpline type')

    def _find_span_linear(self, u: float):
        """Find the span(index) of a single knot(u) over the knot vector(knots) using linear search"""
        span = 0
        while span < len(self.ctrl_points) and self.knots[span] <= u:
            span += 1
        return span - 1

    def __basic(self, degree: int, u: float, index: int):
        """
        Calculate B-Spline basic function N(i, p,u)
        :param degree: degree p
        :param u: input knot u
        :param index: the index of u, i
        :return: basic function evaluated value N(i, p, u)
        """
        if degree == 0:
            return 1.0 if self.knots[index] <= u <= self.knots[index + 1] else 0.0

        c1 = self._basic_coeff(u, self.knots[index], self.knots[index + degree])
        c2 = self._basic_coeff(u, self.knots[index + 1], self.knots[index + 1 + degree])
        return c1 * self.__basic(degree - 1, u, index) + (1.0 - c2) * self.__basic(degree - 1, u, index + 1)

    @staticmethod
    def _basic_coeff(u: float, u0: float, u1: float):
        """
        Evaluate the coefficient (u - u_i) / (u_{i+p} - u_i) = (u - u0) / (u1 - u0), in basic function
        """
        if u0 < u1 and u0 <= u <= u1:
            return (u - u0) / (u1 - u0)
        else:
            return 0


def geomdl_method(degree, ctrl_points):
    """Generate by geomdl package"""
    print(util.Section('B-Spline using geomdl Package'))
    # construct
    curve = BSpline.Curve()
    curve.degree = degree
    curve.ctrlpts = ctrl_points
    curve.knotvector = utilities.generate_knot_vector(degree, len(ctrl_points))
    curve.evaluate(step=0.01)

    print(f'knots length = {len(curve.knotvector)}, knots = {curve.knotvector}')
    print(f'c(0) = {curve.evaluate_single(0)}')
    print(f'c(0.5) = {curve.evaluate_single(0.5)}')
    print(f'c(0.6) = {curve.evaluate_single(0.6)}')
    print(f'c(1.0) = {curve.evaluate_single(1.0)}')

    # plot
    ctrl_plot_points = np.array(ctrl_points)
    curve_points = np.array(curve.evalpts)
    fig = plt.figure('B-Spline using geomdl Package')
    ax = fig.add_subplot(111)
    ax.plot(ctrl_plot_points[:, 0], ctrl_plot_points[:, 1], 'g.-.', label='Control Points')
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'b', label='BSpline Curve')
    ax.grid(True)
    ax.set_title('B-Spline using geomdl Package')
    ax.legend(loc='best')
    plt.show(block=False)


def my_method(degree, ctrl_points):
    print(util.Section('B-Spling using My Method'))
    spline = MyBSpline(degree, ctrl_points)
    print(f'c(0) = {spline.eval(0)}')
    print(f'c(0.5) = {spline.eval(0.5)}')
    print(f'c(0.6) = {spline.eval(0.6)}')
    print(f'c(1.0) = {spline.eval(1.0)}')

    # plot
    u = np.arange(0.0, 1.0, 0.01)
    curve_points = np.zeros((len(u), 2))
    for i, uu in enumerate(u):
        curve_points[i, :] = spline.eval(uu)
    fig = plt.figure('B-Spline using My Method')
    ax = fig.add_subplot(111)
    ax.plot(ctrl_points[:, 0], ctrl_points[:, 1], 'g.-.', label='Control Points')
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'b', label='BSpline Curve')
    ax.grid(True)
    ax.set_title('B-Spline  using My Method')
    ax.legend(loc='best')
    plt.show(block=False)


if __name__ == '__main__':
    ctrl_points = [[0.0, 0.0], [1.0, 5.0], [2.0, 10.0], [5.0, 12.0], [8.0, 10.0], [14.0, -10.0], [16.0, -12.0],
                   [18.0, -10.0], [22.0, 0.0]]
    degree = 3
    print(f'control points = {ctrl_points}')
    print(f'degree = {degree}')

    geomdl_method(degree, ctrl_points)
    my_method(degree, np.array(ctrl_points))

    plt.show(block=True)
