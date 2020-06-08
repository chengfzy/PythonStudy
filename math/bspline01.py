"""
Some study code for BSpline operation

Ref:
    [1] L. Biagiotti and C. Melchiorri, Trajectory Planning for Automatic Machines and Robots. 2008.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import util

# class BSpline:
#     """
#     BSpline
#     """

#     def __init__(self, degree:)


def basic(knots: np.ndarray, degree: int, u: float, index: int):
    """
    Calculate the basic function in BSpline, N(i, p, u)

    Args:
        knots (np.ndarray): Knots vector {u}
        degree (int): Degree p
        u (float): Evaluate input u
        index (int): The index of u, i

    Returns:
        float: Basic function result evaluated at u
    """
    if degree == 0:
        return 1.0 if knots[index] <= u < knots[index + 1] else 0.0

    if knots[index] > u or u >= knots[index + degree + 1]:
        return 0

    c1 = (u - knots[index]) / (knots[index + degree] - knots[index]) if knots[index] <= u < knots[index +
                                                                                                  degree] else 0.0
    c2 = (knots[index + degree + 1] - u) / (
        knots[index + degree + 1] - knots[index + 1]) if knots[index + 1] <= u < knots[index + degree + 1] else 0.0
    return c1 * basic(knots, degree - 1, u, index) + c2 * basic(knots, degree - 1, u, index + 1)


def bspline(knots: np.ndarray, ctrl_points: np.ndarray, degree: int, u: float):
    """
    Evaluate BSpline function at u

    Args:
        knots (np.ndarray): Knots vector {u}
        ctrl_points (np.ndarray): Control points {P}
        degree (int): Degree p
        u (float): Evaluate input u
    Returns:
        np.ndarray: 
    """
    s = np.zeros(ctrl_points.shape[0])
    for m in range(ctrl_points.shape[0]):
        for n in range(ctrl_points.shape[1]):
            s[m] += basic(knots, degree, u, n) * ctrl_points[m, n]
    return s


def cal_basic():
    """
    Calculate the basic function
    """
    print(util.Section('Basic Function'))

    # evaluate of degree 1
    knots = np.array([0.0, 0, 1, 2, 4, 7, 7])
    degree = 1
    u_array = np.arange(knots.min(), knots.max(), 0.01)
    basic_array = np.zeros([7, u_array.shape[0]])
    for n, u in enumerate(u_array):
        for j in range(0, basic_array.shape[0]):
            basic_array[j, n] = basic(knots, degree, u, j)
    # plot
    fig = plt.figure(f'Basic Function of Degree {degree}')
    ax = fig.add_subplot(111)
    ax.grid(True)
    for j in range(0, basic_array.shape[0]):
        ax.plot(u_array, basic_array[j, :], label=f'$B_{{{j}{degree}}}$')
    ax.legend(loc='best')
    ax.set_title(f'Basic function of Degree 1 Defined on u = {knots}')

    # evaluate of degree 3
    knots = np.array([0.0, 0, 0, 0, 1, 2, 4, 7, 7, 7, 7])
    degree = 3
    u_array = np.arange(knots.min(), knots.max(), 0.01)
    basic_array = np.zeros([7, u_array.shape[0]])
    for n, u in enumerate(u_array):
        for j in range(0, basic_array.shape[0]):
            basic_array[j, n] = basic(knots, degree, u, j)
    # plot
    fig = plt.figure(f'Basic Function of Degree {degree}')
    ax = fig.add_subplot(111)
    ax.grid(True)
    for j in range(0, basic_array.shape[0]):
        ax.plot(u_array, basic_array[j, :], label=f'$B_{{{j}{degree}}}$')
    ax.legend(loc='best')
    ax.set_title(f'Basic function of Degree 1 Defined on u = {knots}')


def cal_bspline():
    """
    Calculate the BSpline function
    """
    print(util.Section('BSpline Function'))

    # Evaluation case 1
    knots = np.array([0.0, 0, 0, 0, 1, 2, 4, 7, 7, 7, 7.0])
    ctrl_points = np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, -3, 4, 5, -5, -6]])
    degree = 3
    u_array = np.arange(knots.min(), knots.max(), 0.01)
    bspline_array = np.zeros([ctrl_points.shape[0], u_array.shape[0]])
    for n, u in enumerate(u_array):
        bspline_array[:, n] = bspline(knots, ctrl_points, degree, u)
    # plot
    fig = plt.figure(f'Basic Function of Degree {degree} - Case 1')
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(ctrl_points[0, :], ctrl_points[1, :], 'o--')
    ax.plot(bspline_array[0, :], bspline_array[1, :])
    ax.set_title(f'BSpline function of Degree {degree} Defined on u = {knots}')

    # Evaluation case 2, double the knots in case 1
    knots2 = 2 * knots
    ctrl_points = np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, -3, 4, 5, -5, -6]])
    degree = 3
    u_array2 = np.arange(knots2.min(), knots2.max(), 0.01)
    bspline_array2 = np.zeros([ctrl_points.shape[0], u_array2.shape[0]])
    for n, u in enumerate(u_array2):
        bspline_array2[:, n] = bspline(knots2, ctrl_points, degree, u)
    # plot
    fig = plt.figure(f'Basic Function of Degree {degree} - Double the Knots')
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(u_array, bspline_array[0, :], label='BSpline1[0]')
    ax.plot(u_array, bspline_array[1, :], label='BSpline1[1]')
    ax.plot(u_array2, bspline_array2[0, :], label='BSpline2[0]')
    ax.plot(u_array2, bspline_array2[1, :], label='BSpline2[1]')
    ax.legend(loc='best')
    ax.set_title(f'BSpline function of Degree {degree} Defined on u = {knots}')

    # Evaluation case 3
    knots = np.array([0.0, 0, 0, 0, 2, 2, 2, 7, 7, 7, 7.0])
    ctrl_points = np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, -3, 4, 5, -5, -6]])
    degree = 3
    u_array = np.arange(knots.min(), knots.max(), 0.01)
    bspline_array = np.zeros([ctrl_points.shape[0], u_array.shape[0]])
    for n, u in enumerate(u_array):
        bspline_array[:, n] = bspline(knots, ctrl_points, degree, u)
    # plot
    fig = plt.figure(f'Basic Function of Degree {degree} - Case 3')
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(ctrl_points[0, :], ctrl_points[1, :], 'o--')
    ax.plot(bspline_array[0, :], bspline_array[1, :], label='BSpline1')
    ax.legend(loc='best')
    ax.set_title(f'BSpline function of Degree {degree} Defined on u = {knots}')

    # Evaluation case 4, for different degree
    knots = np.array([0.0, 0, 0, 0, 1, 2, 4, 7, 7, 7, 7])
    ctrl_points = np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, -3, 4, 5, -5, -6]])
    degree = 3
    u_array = np.arange(knots.min(), knots.max(), 0.01)
    bspline_array = np.zeros([ctrl_points.shape[0], u_array.shape[0]])
    bspline_array2 = np.zeros([ctrl_points.shape[0], u_array.shape[0]])
    bspline_array3 = np.zeros([ctrl_points.shape[0], u_array.shape[0]])
    for n, u in enumerate(u_array):
        bspline_array[:, n] = bspline(knots, ctrl_points, degree, u)
        bspline_array2[:, n] = bspline(knots, ctrl_points, degree - 1, u)
        bspline_array3[:, n] = bspline(knots, ctrl_points, degree - 2, u)
    # plot
    fig = plt.figure(f'Basic Function of Degree {degree} to {degree-2} - Case 3')
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(ctrl_points[0, :], ctrl_points[1, :], 'o--')
    ax.plot(bspline_array[0, :], bspline_array[1, :], label='BSpline1')
    ax.plot(bspline_array2[0, :], bspline_array2[1, :], label='BSpline2')
    ax.plot(bspline_array3[0, :], bspline_array3[1, :], label='BSpline3')
    ax.legend(loc='best')
    ax.set_title(f'BSpline function of Degree {degree} to {degree-2} Defined on u = {knots}')


def main():
    # cal_basic()
    cal_bspline()
    plt.show(block=True)


if __name__ == "__main__":
    main()