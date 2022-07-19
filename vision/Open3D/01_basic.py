import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

import util
import open3d as o3d
import numpy as np


def read_and_show():
    """Download PCD data and show"""
    print(util.Section('Download PCD data and show', False))
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


def work_with_numpy_3d():
    """Work with numpy data, nx3"""
    print(util.Section('Work with numpy Data(Nx3)', False))
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print(f'xyz: {xyz}')

    # pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("./data/points3d.pcd", pcd)

    # load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("./data/points3d.pcd")
    o3d.visualization.draw_geometries([pcd_load])


def work_with_numpy_4d():
    """Work with numpy data, nx4"""
    print(util.Section('Work with numpy Data(Nx4)', False))
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyzi = np.zeros((np.size(mesh_x), 4))
    xyzi[:, 0] = np.reshape(mesh_x, -1)
    xyzi[:, 1] = np.reshape(mesh_y, -1)
    xyzi[:, 2] = np.reshape(z_norm, -1)
    xyzi[:, 3] = np.arange(0, xyzi.shape[0])
    print(f'xyzi: {xyzi}')

    # pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.t.geometry.PointCloud()
    pcd.point['positions'] = o3d.core.Tensor(xyzi[:, :3], dtype=o3d.core.Dtype.Float32)
    pcd.point['intensities'] = o3d.core.Tensor(xyzi[:, 3].reshape(xyzi.shape[0], 1), dtype=o3d.core.Dtype.Int16)
    o3d.t.io.write_point_cloud('./data/point4d.pcd', pcd)


if __name__ == '__main__':
    # read_and_show()
    # work_with_numpy_3d()
    work_with_numpy_4d()