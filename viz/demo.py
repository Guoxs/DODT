import sys
import cv2
import numpy as np
import mayavi.mlab as mlab
import  moviepy.editor as mpy

sys.path.append('/home/mooyu/Project/avod/')
sys.path.append('/home/mooyu/Project/avod/wavedata')

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder

from viz.viz_utils import draw_gt_boxes3d

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import tracking_utils
from wavedata.tools.obj_detection.obj_utils import compute_box_corners_3d



def build_dataset(dataset_config):
    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)
    dataset_config.data_split = 'val'
    dataset_config.data_split_dir = 'training'
    dataset_config.has_labels = False
    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []
     # Build the dataset object
    dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config,
                                                     use_defaults=False)
    return dataset

def label_convert(labels, calib):
    length = len(labels)
    corners_3ds = np.zeros((length, 8, 3), dtype=np.float32)
    box_id = []
    for i in range(length):
        gt_box = labels[i]
        boxes3d = compute_box_corners_3d(gt_box).T
        boxes3d = calib.project_rect_to_velo(boxes3d)
        # x, z ,y ==> x, y, z
        corners_3ds[i] = boxes3d[:, [0, 2, 1]]
        box_id.append(gt_box.object_id)
    return corners_3ds, box_id

def get_all_data(dataset, video_id):
    name_list = dataset.get_video_frames(video_id)
    point_clouds = []
    obj_labels = []
    boxes_id = []
    images = []
    calib = calib_utils.read_tracking_calibration(dataset.calib_dir, video_id)

    for name in name_list:
        point_cloud = dataset.kitti_utils.get_raw_point_cloud(dataset.bev_source, name)
        image = cv2.imread(dataset.get_rgb_image_path(name))[..., ::-1]
        obj_label = tracking_utils.read_labels(dataset.label_dir, name)
        obj_label = dataset.kitti_utils.filter_labels(obj_label)
        obj_label, box_id = label_convert(obj_label, calib)

        point_clouds.append(point_cloud)
        images.append(image)
        obj_labels.append(obj_label)
        boxes_id.append(box_id)

    # point cloud align
    max_len = max([pcl.shape[1] for pcl in point_clouds])
    for i in range(len(point_clouds)):
        l = point_clouds[i].shape[1]
        point_clouds[i] = np.pad(point_clouds[i], ((0, 0), (0, max_len - l)),
                                 mode='constant', constant_values=0)

    # labels align
    return point_clouds, obj_labels, boxes_id, images

@mlab.animate(delay=100)
def anim(point_clouds, labels, boxes_id, plt):
    fig = mlab.gcf()
    while True:
        for (pcl, label, ids) in zip(point_clouds, labels, boxes_id):
            print('Updating scene...')
            # draw_gt_boxes3d(label, fig, box_id=ids)
            plt.mlab_source.set(x=pcl[0], y=pcl[1], z=pcl[2], f=pcl[3])
            fig.scene.render()
            yield


config_name = 'pyramid_cars_with_aug_dt_5_tracking_test'
# Read the config from the config folder
experiment_config_path = avod.root_dir() + '/configs/' + \
                         config_name + '.config'

model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            experiment_config_path, is_training=False)

# change datasets dir to local
dataset_config.dataset_dir = '/media/mooyu/Guoxs_Data/Datasets/' \
                             '3D_Object_Tracking_Evaluation_2012'

video_id = 4
dataset = build_dataset(dataset_config)
point_clouds, labels, boxes_id, images = get_all_data(dataset, video_id)

fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))

# draw lidar
plt = mlab.points3d(point_clouds[0][0],
                    point_clouds[0][1],
                    point_clouds[0][2],
                    point_clouds[0][3],
                    color=None,
                    mode='point',
                    colormap='gnuplot',
                    scale_factor=1,
                    figure=fig)

# draw boxes3d
# draw_gt_boxes3d(labels[0], fig, box_id=boxes_id[0])
# draw origin
mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.05)

anim(point_clouds, labels, boxes_id, plt)

mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991],
          distance=62.0, figure=fig)
mlab.show()