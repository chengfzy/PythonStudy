"""
Try NuScenes Dataset
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

# below code is used to fix "mix incompatible Qt library" bugs
import matplotlib.pyplot as plt
import cv2

import util
from nuscenes.nuscenes import NuScenes


def look_at_dataset(nusc: NuScenes):
    print(util.Title('A Look at the Dataset'))
    print(util.Section('Scene'))
    nusc.list_scenes()
    my_scene = nusc.scene[0]
    print(f'my scene: {my_scene}')

    print(util.Section('Sample'))
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    print(f'my_sample: {my_sample}')
    print(util.Paragraph('List Sample'))
    nusc.list_sample(my_sample['token'])

    print(util.Section('Sample Data'))
    print(f'data: {my_sample["data"]}')
    sensor = 'CAM_FRONT'
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
    print(f'cam front data: {cam_front_data}')
    nusc.render_sample_data(cam_front_data['token'])

    print(util.Section('Sample Annotation'))
    my_annotation_token = my_sample['anns'][18]
    my_annotation_metadata = nusc.get('sample_annotation', my_annotation_token)
    print(f'my annotation metadata: {my_annotation_metadata}')
    nusc.render_annotation(my_annotation_token)

    print(util.Section('Instance'))
    my_instance = nusc.instance[599]
    print(f'my instance: {my_instance}')
    instance_token = my_instance['token']
    nusc.render_instance(instance_token, extra_info=True)
    print('First annotated sample of this instance')
    nusc.render_annotation(my_instance['first_annotation_token'], extra_info=True)
    print('Last annotated sample of this instance')
    nusc.render_annotation(my_instance['last_annotation_token'], extra_info=True)

    print(util.Section('Category'))
    nusc.list_categories()
    print(f'category: {nusc.category[9]}')

    print(util.Section('Attribute'))
    nusc.list_attributes()
    my_instance = nusc.instance[27]
    first_token = my_instance['first_annotation_token']
    last_token = my_instance['last_annotation_token']
    nbr_samples = my_instance['nbr_annotations']
    current_token = first_token
    i = 0
    found_change = False
    while current_token != last_token:
        current_ann = nusc.get('sample_annotation', current_token)
        current_attr = nusc.get('attribute', current_ann['attribute_tokens'][0])['name']
        if i == 0:
            pass
        elif current_attr != last_attr:
            print(f'Change from "{last_attr}" to "{current_attr}" at timestamp {i}/{nbr_samples} annotated timestamps')
            found_change = True
        next_token = current_ann['next']
        current_token = next_token
        last_attr = current_attr
        i += 1

    print(util.Section('Visibility'))
    print(f'visibility: {nusc.visibility}')
    # look at an example sample_annotation with 80-100% visibility
    ann_token = 'a7d0722bce164f88adf03ada491ea0ba'
    visibility_token = nusc.get('sample_annotation', ann_token)['visibility_token']
    print(f'visibility: {nusc.get("visibility", visibility_token)}')
    nusc.render_annotation(ann_token)
    # look at an example sample_annotation with 0-40% visibility
    ann_token = '9f450bf6b7454551bbbc9a4c6e74ef2e'
    visibility_token = nusc.get('sample_annotation', ann_token)['visibility_token']
    print(f'visibility: {nusc.get("visibility", visibility_token)}')
    nusc.render_annotation(ann_token)

    print(util.Section('Sensor'))
    print(f'sensor: {nusc.sensor}')
    print(f'sensor data: {nusc.sample_data[10]}')

    print(util.Section('Calibrated Sensor'))
    print(f'calibrated sensor: {nusc.calibrated_sensor[0]}')

    print(util.Section('Ego Pose'))
    print(f'ego pose: {nusc.ego_pose[0]}')

    print(util.Section('Log'))
    print(f'number of logs in our loaded database: {len(nusc.log)}')
    print(f'log: {nusc.log[0]}')

    print(util.Section('Map'))
    print(f'There are {len(nusc.map)} maps masks in the loaded dataset')
    print(f'map: {nusc.map[0]}')


def basic(nusc: NuScenes):
    print(util.Title('nuScenes Basics'))
    print(f'category: {nusc.category[0]}')

    cat_token = nusc.category[0]['token']
    print(f'token: {cat_token}')
    print(f'record from token: {nusc.get("category", cat_token)}')
    print(f'sample annotation: {nusc.sample_annotation[0]}')
    print(f'visibility: {nusc.get("visibility", nusc.sample_annotation[0]["visibility_token"])}')
    one_instance = nusc.get("instance", nusc.sample_annotation[0]["instance_token"])
    print(f'instance: {one_instance}')

    print(util.Section('Recover All Sample Annotations for a Particular Object Instance'))
    # method 01
    print(util.SubSection('Method 01'))
    ann_tokens = nusc.field2token('sample_annotation', 'instance_token', one_instance['token'])
    ann_tokens_field2token = set(ann_tokens)
    print(f'ann tokens: {ann_tokens_field2token}')

    # method 02, traverse all annotations of the instance
    print(util.SubSection('Method 02'))
    ann_record = nusc.get('sample_annotation', one_instance['first_annotation_token'])
    print(f'ann record: {ann_record}')
    ann_tokens_traverse = set()
    ann_tokens_traverse.add(ann_record['token'])
    while not ann_record['next'] == "":
        ann_record = nusc.get('sample_annotation', ann_record['next'])
        ann_tokens_traverse.add(ann_record['token'])
    print(f'ann tokens: {ann_tokens_traverse}')


def reverse_index_and_shortcuts(nusc: NuScenes):
    print(util.Title('Reverse indexing and short-cuts'))
    print(util.Section('Shortcuts'))
    cat_name = nusc.sample_annotation[0]['category_name']
    print(f'category name using shortcut: {cat_name}')
    print('not using shortcut')
    ann_rec = nusc.sample_annotation[0]
    inst_rec = nusc.get('instance', ann_rec['instance_token'])
    cat_rec = nusc.get('category', inst_rec['category_token'])
    print(f'category name not using shortcut: {cat_rec["name"]}')


def data_visualization(nusc: NuScenes):
    print(util.Title('Data Visualizations'))
    print(util.Section('List Method'))
    print(util.Paragraph('List Categories'))
    print(f'{nusc.list_categories()}')
    print(util.Paragraph('List Attributes'))
    print(f'{nusc.list_attributes()}')
    print(util.Paragraph('List Scenes'))
    print(f'{nusc.list_scenes()}')

    print(util.Section('Render'))
    my_sample = nusc.sample[10]
    nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', verbose=False)
    nusc.render_pointcloud_in_image(my_sample['token'],
                                    pointsensor_channel='LIDAR_TOP',
                                    render_intensity=True,
                                    verbose=False)
    nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='RADAR_FRONT', verbose=False)

    print(util.Section('Plot Sample Data'))
    my_sample = nusc.sample[20]
    nusc.render_sample(my_sample['token'], verbose=False)
    # only particular sensor
    nusc.render_sample_data(my_sample['data']['CAM_FRONT'], verbose=False)
    # aggregate the point clouds from multiple sweeps to get a denser point cloud
    nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True, verbose=False)
    nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True, verbose=False)

    print(util.Section('Disable RADAR Filter'))
    from nuscenes.utils.data_classes import RadarPointCloud
    RadarPointCloud.disable_filters()
    nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True, verbose=False)
    RadarPointCloud.default_filters()

    print(util.Section('Render as a Video'))
    my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0]
    nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')
    # render all camera
    nusc.render_scene(my_scene_token)
    # visualize all scenes on the map for a particular location
    nusc.render_egoposes_on_map(log_location='singapore-onenorth')


def get_calibration_params(nusc: NuScenes):
    print(util.Section('Get Calibration Parameters'))
    print(util.SubSection('Sensors'))
    print(f'sensors: {nusc.sensor}')
    print(util.SubSection('Calibration Parameters'))
    print(f'calibrated sensor: {nusc.calibrated_sensor}')


if __name__ == '__main__':
    print(util.Title('Tutorial'))
    print('Ref: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_tutorial.ipynb')

    nusc = NuScenes(version='v1.0-mini', dataroot='/home/jeffery/Documents/DataSet/nuScenes/v1.0-mini', verbose=True)
    # look_at_dataset(nusc)
    # basic(nusc)
    # reverse_index_and_shortcuts(nusc)
    # data_visualization(nusc)
    get_calibration_params(nusc)

    plt.show(block=True)
