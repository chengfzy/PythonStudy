import numpy as np
import util
import matplotlib.pyplot as plt
import scipy.optimize as opt
from mpl_toolkits.mplot3d import Axes3D


def fit_line_svd(data):
    """
    Fit 3D line using SVD method, [x, y, z] = [a * t + x0, b * t + y0, c * t + z0]
    :param data: 3D points, N x 3 array
    :return: 2 x 3 matrix, [a, b, c; x0, y0, z0]
    """
    xyz0 = data.mean(axis=0)
    centered_data = data - xyz0
    u, s, vh = np.linalg.svd(centered_data)
    # the vector corresponding to max eigenvalue will be the line direction
    return vh[0, :], xyz0


def fit_line_opt(data):
    """
    Fit 3D line using optimization method, [x, y, z] = [a * t + x0, b * t + y0, c * t + z0]
    :param data: 3D points, N x 3 array
    :return: 2 x 3 matrix, [a, b, c; x0, y0, z0]
    """

    N = data.shape[0]

    def cost_func(x0):
        print(f'x0 = {x0}')
        abc = x0[:3]
        xyz0 = x0[3:]
        cost = np.zeros((N, 1))
        for n in range(N):
            m = data[n, :] - xyz0
            cost[n] = np.linalg.norm(np.cross(m, abc)) / np.linalg.norm(abc)

    x0 = np.array([0.1, 0.2, 0.3, 4.0, 5.0, 6.0])
    res = opt.minimize(cost_func, x0)


def line_fit_example():
    """example for 3D line fitting"""
    print(util.Section('3D Line Fitting'))

    # generate 3D points data, [x, y, z] = [a * t + x0, b * t + y0, c * t + z0]
    print(util.Section('Generate 3D Points Data'))
    abc = np.array([0.5, 0.6, 0.7])
    xyz0 = np.array([1.0, 2.0, 3.0])
    t = np.arange(-3.0, 3.0, 1)
    xyz = np.zeros((t.shape[0], 3))
    for n in range(t.shape[0]):
        xyz[n, :] = t[n] * abc + xyz0
    print(f'abc = {abc}, xyz0 = {xyz0}')

    # 3D line fitting using SVD
    print(util.SubSection('Fitting use SVD'))
    abc_svd, xyz0_svd = fit_line_svd(xyz)
    print(f'SVD: abc = {abc_svd}, xyz0 = {xyz0_svd}')

    # 3D line fitting using optimization
    fit_line_opt(xyz)

    # generate plot data
    t_plot = np.arange(-5.0, 5.0, 0.01)
    xyz_svd = np.zeros((t_plot.shape[0], 3))
    for n in range(t_plot.shape[0]):
        xyz_svd[n, :] = t_plot[n] * abc_svd + xyz0_svd
    # plot
    fig = plt.figure('3D Line Fitting')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'k.', label='Measurement')
    ax.plot(xyz_svd[:, 0], xyz_svd[:, 1], xyz_svd[:, 2], 'r', label='SVD Fitting')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Line Fitting')
    ax.legend()
    plt.show(block=True)


if __name__ == '__main__':
    line_fit_example()
