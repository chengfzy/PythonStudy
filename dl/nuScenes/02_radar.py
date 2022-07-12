import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

# below code is used to fix "mix incompatible Qt library" bugs
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.patches import Ellipse, Arrow

import argparse
import logging
import coloredlogs
import numpy as np
import util
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


class NuSceneAnalysis:

    def __init__(self, dataroot: Path, version='v1.0-mini') -> None:
        logging.info(f'data root: {dataroot}')
        logging.info(f'version: {version}')
        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)

    def test(self):
        """
        Test code
        """
        # self.render_test()
        self.my_render_test(scene_name='scene-0103')

        plt.show(block=True)

    def render_test(self):
        """
        Render using nuScenes API
        """
        # get sample
        my_scene_tokens = self.nusc.field2token('scene', 'name', 'scene-0061')[0]  # only one element
        my_scene = self.nusc.get('scene', my_scene_tokens)
        my_sample_token = my_scene['first_sample_token']
        my_sample = self.nusc.get('sample', my_sample_token)
        # sensor channel
        cam_channel = 'CAM_FRONT'
        lidar_channel = 'LIDAR_TOP'
        radar_channel = 'RADAR_FRONT'

        logging.info(util.Section('Render Camera as Video', False))
        self.render_camera_video(my_scene_tokens, cam_channel)
        self.render_camera_video_default(my_scene_tokens, cam_channel)

        logging.info(util.Section('Render Point Cloud in Image', False))
        self.render_point_cloud_in_image(my_sample_token, cam_channel, lidar_channel)
        self.render_point_cloud_in_image(my_sample_token, cam_channel, radar_channel)

        logging.info(util.Section('Render Sample Data', False))
        self.render_sample_data_default(my_sample['data'][cam_channel])
        self.render_sample_data_default(my_sample['data'][lidar_channel])
        self.render_sample_data_default(my_sample['data'][radar_channel])

    def my_render_test(self, scene_name='scene-0061'):
        """
        Render using plot method written by me
        """
        logging.info(util.Section('Render RADAR Data Using My Method', False))
        logging.info('disable RADAR filters')
        RadarPointCloud.disable_filters()

        cam_channel = 'CAM_FRONT'
        radar_channel = 'RADAR_FRONT'
        scene_token = self.nusc.field2token('scene', 'name', scene_name)[0]
        scene = self.nusc.get('scene', scene_token)
        fig = None
        ax = None
        sample_token = scene['first_sample_token']
        while sample_token != scene['last_sample_token']:
            sample = self.nusc.get('sample', sample_token)
            # render image
            cam_sample_data = self.nusc.get('sample_data', sample['data'][cam_channel])
            cam_path = Path(self.nusc.dataroot) / cam_sample_data['filename']
            logging.info(f'read image: {cam_path}')
            cam_data = cv.imread(str(cam_path), cv.IMREAD_UNCHANGED)
            cam_data = cv.resize(cam_data, dsize=None, fx=0.5, fy=0.5)
            cv.imshow(f'Camera {cam_channel}', cam_data)

            # render RADAR
            fig, ax = self.render_radar_sample(fig, ax, sample['data'][radar_channel])

            # render RADAR data using default API
            # self.render_sample_data_default(sample['data'][radar_channel])

            # got to nex
            cv.waitKey(0)
            sample_token = sample['next']

        cv.waitKey()
        plt.show(block=True)

    def render_camera_video_default(self, scene_token: str, channel='CAM_FRONT'):
        """Render camera data as video using default method"""
        self.nusc.render_scene_channel(scene_token, channel)

    def render_camera_video(self, scene_token: str, channel='CAM_FRONT') -> None:
        """
        Render camera data as video

        Args:
            scene_tokens (str): _description_
        """
        my_scene = self.nusc.get('scene', scene_token)
        sample_token = my_scene['first_sample_token']
        while sample_token != my_scene['last_sample_token']:
            my_sample = self.nusc.get('sample', sample_token)
            cam_sample_data = self.nusc.get('sample_data', my_sample['data'][channel])
            cam_path = Path(self.nusc.dataroot) / cam_sample_data['filename']
            logging.info(f'read image: {cam_path}')
            cam_data = cv.imread(str(cam_path), cv.IMREAD_UNCHANGED)
            cv.imshow('Camera', cam_data)
            cv.waitKey(100)
            sample_token = my_sample['next']
        cv.waitKey()

    def render_point_cloud_in_image(self,
                                    sample_token,
                                    camera_channel='CAM_FRONT',
                                    point_channel='RADAR_FRONT') -> None:
        self.nusc.render_pointcloud_in_image(sample_token,
                                             pointsensor_channel=point_channel,
                                             camera_channel=camera_channel,
                                             verbose=False)

    def render_sample_data_default(self, sample_data_token: str):
        self.nusc.render_sample_data(sample_data_token,
                                     with_anns=True,
                                     nsweeps=1,
                                     show_lidarseg=True,
                                     show_lidarseg_legend=True,
                                     underlay_map=True,
                                     use_flat_vehicle_coordinates=True,
                                     verbose=False)

    def render_radar_sample(self, fig=None, ax=None, sample_data_token: str = None):
        # read radar data
        radar_sample_data = self.nusc.get('sample_data', sample_data_token)
        file_path = Path(self.nusc.dataroot) / radar_sample_data['filename']
        logging.info(f'read RADAR: {file_path}')
        radar_data = RadarPointCloud.from_file(str(file_path), RadarPointCloud.invalid_states,
                                               RadarPointCloud.dynprop_states, RadarPointCloud.ambig_states)

        # obtain data
        # logging.info(f'radar point shape = {radar_data.points.shape}')
        pos = radar_data.points[:2, :].transpose()  # pos (x, y)
        id = radar_data.points[4, :].transpose()  # id
        vel = radar_data.points[6:8, :].transpose()  # vel, (x, y)
        pos_rms = radar_data.points[12:14, :].transpose()  # pos rms, (x, y)

        # plot
        if fig is None:
            plt.interactive(True)
            fig = plt.figure('RDDAR Data', figsize=(8, 6))
            ax = fig.subplots(1, 1)
        else:
            ax.clear()
        ax.plot([0], [0], 'rx', markersize=10)  # origin
        ax.plot(pos[:, 0], pos[:, 1], 'r.', label='Cluster')
        self.__plt_confidence_ellipse(ax, pos, pos_rms, scale=0.1)  # confidence ellipse
        self.__plot_vel_arrow(ax, pos, vel, scale=0.5)  # vel arrow
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'RADAR Data t = {radar_sample_data["timestamp"]*1e-6:.5f} s')
        ax.grid(True)
        # ax.set_xlim(-10, 50)
        # ax.set_ylim(-10, 50)
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        fig.canvas.flush_events()  # update in background

        return fig, ax

    def __plt_confidence_ellipse(self, ax, pos: np.ndarray, pos_rms: np.ndarray, scale=1.0):
        num = pos.shape[0]
        if num != pos_rms.shape[0]:
            raise ValueError('pos and rms data should be the same size')

        for n in range(num):
            p = pos[n, :]
            rms = pos_rms[n, :]
            ellipse = Ellipse((p[0], p[1]),
                              rms[0] * scale,
                              rms[1] * scale,
                              facecolor=(0, 1, 0, 0.8),
                              edgecolor=(0, 0, 0, 0.5))
            ax.add_patch(ellipse)

    def __plot_vel_arrow(self, ax, pos: np.ndarray, vel: np.ndarray, scale=1.0):
        num = pos.shape[0]
        if num != vel.shape[0]:
            raise ValueError('pos and vel data should be the same size')

        for n in range(num):
            p = pos[n, :]
            v = vel[n, :]
            arrow = Arrow(p[0], p[1], v[0] * scale, v[1] * scale)
            ax.add_patch(arrow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nuScenes RADAR Analysis')
    parser.add_argument('folder', type=str, help='data root folder')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='data version')
    args = parser.parse_args()
    print(args)

    # config logging
    logging.basicConfig(level=logging.INFO)
    coloredlogs.install(fmt="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s")

    analysis = NuSceneAnalysis(dataroot=args.folder, version=args.version)
    analysis.test()
