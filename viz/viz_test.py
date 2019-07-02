import cv2
import numpy as np

import sys
sys.path.append('/home/mooyu/Project/avod/')
sys.path.append('/home/mooyu/Project/avod/wavedata')

import avod.tests as tests
import mayavi.mlab as mlab
from avod.builders.dataset_builder import DatasetBuilder
from viz.viz_utils import draw_lidar_simple
from wavedata.tools.obj_detection import tracking_utils
from viz.viz_func import draw_lidar_and_boxes,draw_lidar_and_boxes_in_camera_view

def test_oxts_coordinate_transform():
    kitti_dir = tests.test_path() + "/datasets/Kitti/tracking"
    dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_UNITTEST)
    # Overwrite config values
    dataset_config.data_split = 'trainval'
    dataset_config.dataset_dir = kitti_dir
    dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config)
    sample_names = ['020040', '020045']
    raw_point_cloud = []
    raw_label = []
    calibs = []

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    for i in range(2):
        point_cloud = dataset.kitti_utils.get_raw_point_cloud(
            dataset.bev_source, sample_names[i])
        obj_label = tracking_utils.read_labels(dataset.label_dir, sample_names[i])
        obj_label = dataset.kitti_utils.filter_labels(obj_label)
        calib = dataset.kitti_utils.get_calib(dataset.bev_source, sample_names[i])
        raw_point_cloud.append(point_cloud)
        raw_label.append(obj_label)
        calibs.append(calib)
        fig = draw_lidar_and_boxes(point_cloud, obj_label, calib, fig)
    mlab.show()
    input()

    cv_bgr_image = [cv2.imread(dataset.get_rgb_image_path(name)) for name in sample_names]
    rgb_image = [image[..., :: -1] for image in cv_bgr_image]
    image_shape = [img.shape[0:2] for img in rgb_image]

    ct_fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    ct_obj_label = dataset.label_transform(raw_label, sample_names)
    ct_point_cloud = dataset.point_cloud_transform(raw_point_cloud, sample_names)
    for i in range(2):
        ct_fig = draw_lidar_and_boxes(ct_point_cloud[i], ct_obj_label[i], calibs[i], ct_fig)
    mlab.show()
    input()

    ct_fig2 = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    for i in range(2):
        ct_point_cloud[i] = dataset.kitti_utils.transfer_lidar_to_camera_view(
            dataset.bev_source, sample_names[i], ct_point_cloud[i], image_shape[i])
        ct_fig2 = draw_lidar_and_boxes_in_camera_view(ct_point_cloud[i], ct_obj_label[i], ct_fig2)
    mlab.show()
    input()


def test_normal():
    root_dir = '/media/mooyu/Guoxs_Data/Datasets/3D_Object_Tracking_Evaluation_2012'
    lidar_file1 = root_dir + '/training/velodyne/0000/000000.bin'
    lidar_file2 = root_dir + '/training/velodyne/0000/000005.bin'
    lidar1 = np.fromfile(lidar_file1,dtype=np.single).reshape(-1,4)
    lidar2 = np.fromfile(lidar_file2, dtype=np.single).reshape(-1, 4)
    merge_lidar = np.concatenate([lidar1[:, :3], lidar2[:, :3]], axis=0)


    kitti_dir = tests.test_path() + "/datasets/Kitti/tracking"
    dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_UNITTEST)
    # Overwrite config values
    dataset_config.data_split = 'trainval'
    dataset_config.dataset_dir = kitti_dir
    dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config)
    sample_names = ['000000', '000005']
    ct_lidar = dataset.point_cloud_transform([lidar1[:, :3].T, lidar2[:, :3].T], sample_names)
    merge_ct_lidar = np.concatenate(ct_lidar, axis=1)
    draw_lidar_simple((merge_ct_lidar.T))
    input()


# test_normal()

test_oxts_coordinate_transform()