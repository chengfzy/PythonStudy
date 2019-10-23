"""
Use SciPy to solve bundle adjustment

Ref: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""

from __future__ import print_function
import urllib3
import shutil
import bz2
import numpy as np
import util
import os
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares


class BundleAdjustment:
    def __init__(self):
        self.__file_name = "problem-49-7776-pre.txt.bz2"
        self.__data_folder = '../data/'
        self.data_file = os.path.join(self.__data_folder, self.__file_name)
        pass

    def solve(self):
        self.__download_data()
        self.__read_bal_data(self.data_file)

        x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        f0 = self.cost_fun(x0, self.n_cameras, self.n_points, self.camera_indices, self.point_indices, self.points_2d)
        # plot
        plt.plot(f0)

        A = self.bundle_adjustment_sparsity(self.n_cameras, self.n_points, self.camera_indices, self.point_indices)
        t0 = time.time()
        res = least_squares(self.cost_fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(
            self.n_cameras, self.n_points, self.camera_indices, self.point_indices, self.points_2d))
        t1 = time.time()
        print(f'Optimization took {t1 - t0: .3f} seconds')

        plt.plot(res.fun)
        plt.show(block=True)

    @staticmethod
    def rotate(points, rot_vecs):
        """
        Rotate points by given rotation vectors, use Rodrigues' rotation formula
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def project(self, points, camera_params):
        """Convert 3D points to 2D by projection onto images"""
        points_proj = self.rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        f = camera_params[:, 6]
        k1 = camera_params[:, 7]
        k2 = camera_params[:, 8]
        n = np.sum(points_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        points_proj *= (r * f)[:, np.newaxis]
        return points_proj

    def cost_fun(self, params, n_cameras, n_points, camera_indices, point_indices, point_2d):
        """
        Compute residuals.
        `params` contains camera parameters and 3D coordinates
        """
        camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = params[n_cameras * 9:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - point_2d).ravel()

    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 9 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(9):
            A[2 * i, camera_indices * 9 + s] = 1
            A[2 * i + 1, camera_indices * 9 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

        return A

    def __download_data(self):
        """download data"""
        base_url = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
        url = base_url + self.__file_name

        if not os.path.isfile(self.data_file):
            c = urllib3.PoolManager()
            with c.request('GET', url, preload_content=False) as resp, open(self.data_file, 'wb') as f:
                shutil.copyfileobj(resp, f)

    def __read_bal_data(self, file_name):
        print(util.Section('Read Data from File'))
        with bz2.open(file_name, 'rt') as file:
            self.n_cameras, self.n_points, n_observations = map(int, file.readline().split())
            self.camera_indices = np.empty(n_observations, dtype=int)
            self.point_indices = np.empty(n_observations, dtype=int)
            self.points_2d = np.empty((n_observations, 2))

            for i in range(n_observations):
                camera_idx, point_idx, x, y = file.readline().split()
                self.camera_indices[i] = int(camera_idx)
                self.point_indices[i] = int(point_idx)
                self.points_2d[i] = [float(x), float(y)]

            # camera parameters: R, t, f, k1, k2. and R are specified as Rodrigues' vector
            self.camera_params = np.empty(self.n_cameras * 9)
            for i in range(self.n_cameras * 9):
                self.camera_params[i] = float(file.readline())
            self.camera_params = self.camera_params.reshape((self.n_cameras, -1))

            self.points_3d = np.empty(self.n_points * 3)
            for i in range(self.n_points * 3):
                self.points_3d[i] = float(file.readline())
            self.points_3d = self.points_3d.reshape((self.n_points, -1))

        print(f'n_cameras: {self.n_cameras}')
        print(f'n_points: {self.n_points}')
        print(f'Total number of parameters: {9 * self.n_cameras + 3 * self.n_points}')
        print(f'Total number of residuals: {2 * self.points_2d.shape[0]}')


if __name__ == '__main__':
    bundleAdjustment = BundleAdjustment()
    bundleAdjustment.solve()
