import sys
import numpy as np
sys.path.append('/home/mooyu/Project/avod/')
sys.path.append('/home/mooyu/Project/avod/wavedata')

from avod.core import constants
from avod.protos import pipeline_pb2

import mayavi.mlab as mlab
from avod.builders.dataset_builder import DatasetBuilder
from viz.viz_utils import draw_lidar_simple, draw_gt_boxes3d

def build_datasets():
    pipeline_config = pipeline_pb2.NetworkPipelineConfig()
    dataset_config = pipeline_config.dataset_config

    dataset_config.MergeFrom(DatasetBuilder.KITTI_TRACKING_UNITTEST)
    dataset = DatasetBuilder.build_kitti_tracking_stack_dataset(dataset_config)

    return dataset

def compute_corner_box3d(gt_labels):
    length = len(gt_labels)
    corners_3ds = np.zeros((length, 8, 3), dtype=np.float32)
    for i in range(length):
        label = gt_labels[i]
        # Compute rotational matrix
        rot = np.array([[+np.cos(label[6]), 0, +np.sin(label[6])],
                        [0, 1, 0],
                        [-np.sin(label[6]), 0, +np.cos(label[6])]])

        l = label[3]
        w = label[4]
        h = label[5]

        # 3D BB corners
        x_corners = np.array(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
        z_corners = np.array(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

        corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

        corners_3d[0, :] = corners_3d[0, :] + label[0]
        corners_3d[1, :] = corners_3d[1, :] + label[1]
        corners_3d[2, :] = corners_3d[2, :] + label[2]

        corners_3ds[i] = corners_3d.T

    return corners_3ds


dataset = build_datasets()
print(dataset.sample_names)
samples = dataset.load_samples([7])
integrated_pc = samples[0][constants.KEY_INTEGRATED_POINT_CLOUD]
integrated_boxes3d = samples[0][constants.KEY_INTEGRATED_LABEL_BOX_3D]
sepaprated_boxes3d = samples[0][constants.KEY_LABEL_BOXES_3D]

integrated_box_id = integrated_boxes3d[:,-1]
separated_box_id = sepaprated_boxes3d[:,-1]

integrated_corner_box3d = compute_corner_box3d(integrated_boxes3d)
separated_corner_box3d = compute_corner_box3d(sepaprated_boxes3d)

fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
fig = draw_lidar_simple(integrated_pc, fig=fig)
fig = draw_gt_boxes3d(integrated_corner_box3d, fig, box_id=integrated_box_id, color=(0,1,0))
fig = draw_gt_boxes3d(separated_corner_box3d, fig, draw_text=False, color=(1,1,1))
mlab.show()
input()