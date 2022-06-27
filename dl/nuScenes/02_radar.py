import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

# below code is used to fix "mix incompatible Qt library" bugs
import matplotlib.pyplot as plt
import cv2 as cv
import logging
import coloredlogs

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

        logging.info(util.Section('render point cloud in image', False))
        self.render_point_cloud_in_image(my_sample_token, cam_channel, lidar_channel)
        self.render_point_cloud_in_image(my_sample_token, cam_channel, radar_channel)

        logging.info(util.Section('render sample data', False))
        self.render_sample_data_default(my_sample['data'][cam_channel])
        self.render_sample_data_default(my_sample['data'][lidar_channel])
        self.render_sample_data_default(my_sample['data'][radar_channel])

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

    def render_radar_sample(self, token: str):
        pass


if __name__ == '__main__':
    # config logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    coloredlogs.install(fmt="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s")

    analysis = NuSceneAnalysis(dataroot='/home/jeffery/Documents/DataSet/nuScenes/v1.0-mini', version='v1.0-mini')
    analysis.test()
