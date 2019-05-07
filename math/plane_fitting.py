import numpy as np
import common.debug_info as debug_info
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fit_plane_svd(data):
    """
    Fit plane using SVD, ax + by + cz + d = 0
    :param data: 3D points
    :return: normalized direction [a, b, c]
    """
    # if don't remove center, don't changed the value (a,b,c)
    avg = data.mean(axis=0)
    centered_data = data - avg
    cal_data = np.c_[centered_data, np.ones(len(data))]
    u, s, vh = np.linalg.svd(cal_data)

    # params
    abc = vh[-1, :-1]
    abc = abc / np.linalg.norm(abc)
    d = - np.dot(abc, avg)
    return np.hstack((abc, d))


def plane_fitting_example():
    """example for 3D line fitting"""
    print(debug_info.section('3D Plane Fitting'))

    # generate 3D points data, ax + by + cz + d = 0
    print(debug_info.section('Generate 3D Points Data'))
    # generate simulated data
    params = np.array([1.0, 2.0, 3.0, 4.0])
    x, y = np.meshgrid(np.arange(-10, 10, 1), np.arange(-10, 10, 1))
    z = -(params[0] * x + params[1] * y + params[3]) / params[2]
    plane_data = np.c_[x.flatten(), y.flatten(), z.flatten()]
    print(f'abcd = {params}')
    print(f'normalized abcd = {params / np.linalg.norm(params)}')

    # plane fitting
    print(debug_info.section('Generate 3D Points Data'))
    params_svd = fit_plane_svd(plane_data)
    print(f'SVD, abcd = {params_svd}')
    print(f'SVD, normalized abcd = {params_svd / np.linalg.norm(params_svd)}')

    # generate plot data
    x_plot, y_plot = np.meshgrid(np.arange(-10, 10, 0.01), np.arange(-10, 10, 0.01))
    z_plot = -(params_svd[0] * x_plot + params_svd[1] * y_plot + params_svd[3]) / params_svd[2]
    # plot
    fig = plt.figure('3D Plane Fitting')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x.flatten(), y.flatten(), z.flatten(), 'k.', label='Measurement')
    ax.plot_surface(x_plot, y_plot, z_plot, alpha=0.2, color='r', label='SVD Fitting')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plane Fitting')
    plt.show(block=True)


if __name__ == '__main__':
    plane_fitting_example()
