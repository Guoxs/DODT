import cv2
import os
import numpy as np

import tensorflow as tf
from sklearn.cluster import DBSCAN

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import anchor_projector, box_3d_encoder
from wavedata.tools.obj_detection import tracking_utils
from wavedata.tools.obj_detection.obj_utils import compute_box_corners_3d

def draw_bounding_box2d(img, box2d):
    color = (1,0,0)
    box2d = box2d.astype(np.uint8)
    for (i,j) in zip([0,1,2,3], [1,2,3,0]):
        cv2.line(img, (box2d[i,0], box2d[i,1]),
                 (box2d[j,0], box2d[j,1]), color, 1)
    return img


def build_dataset(eval_config, dataset_config):
    # Parse eval config
    eval_mode = eval_config.eval_mode

    # Parse dataset config
    data_split = dataset_config.data_split
    if data_split == 'train':
        dataset_config.data_split_dir = 'training'
        dataset_config.has_labels = True

    elif data_split.startswith('val'):
        dataset_config.data_split_dir = 'training'

        # Don't load labels for val split when running in test mode
        if eval_mode == 'val':
            dataset_config.has_labels = True
        elif eval_mode == 'test':
            dataset_config.has_labels = False

    elif data_split == 'test':
        dataset_config.data_split_dir = 'testing'
        dataset_config.has_labels = False

    else:
        raise ValueError('Invalid data split', data_split)

    # Convert to object to overwrite repeated fields
    dataset_config = config_builder.proto_to_obj(dataset_config)

    # Remove augmentation during evaluation
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_tracking_stack_dataset(
        dataset_config, use_defaults=False)
    return dataset


def draw_anchors(dataset, pred_path, cpkt_idx, sample_names):
    # info
    image_shape = [360, 1200]
    area_extents = np.array([[-40, 40], [-5, 5], [0, 70]])
    ground_plane = np.array([0, -1, 0, 1.65])

    first_name = sample_names.split('_')[0]
    # get point
    point_cloud = dataset.kitti_utils.get_raw_point_cloud(
        dataset.bev_source, first_name)

    point_cloud = dataset.kitti_utils.transfer_lidar_to_camera_view(
        dataset.bev_source, first_name, point_cloud, image_shape)

    # convert point cloud to bev map and store
    bev_images = dataset.kitti_utils.create_bev_maps(point_cloud, ground_plane)
    height_maps = bev_images.get('height_maps')
    density_map = bev_images.get('density_map')
    bev_input = np.dstack((*height_maps, density_map))
    bev_input = np.mean(bev_input, axis=2)
    bev_input = bev_input * 255.0
    cv2.imwrite(first_name + '.png', bev_input)

    # get label
    obj_label = tracking_utils.read_labels(dataset.label_dir, first_name)
    obj_label = dataset.kitti_utils.filter_labels(obj_label)

    # project label to bev
    boxes3d = [box_3d_encoder.object_label_to_box_3d(obj) for obj in obj_label]
    anchors = box_3d_encoder.box_3d_to_anchor(boxes3d, ortho_rotate=True)
    _, bev_norm_boxes = \
        anchor_projector.project_to_bev(anchors, area_extents[[0, 2]])

    bev_boxes = bev_norm_boxes * [800, 700, 800, 700]
    bev_boxes = bev_boxes.astype(np.int32)

    # draw bev_boxes in img
    img = cv2.imread(first_name + '.png')
    for box in bev_boxes:
        cv2.rectangle(img, (box[0],box[1]), (box[2], box[3]), (0,0,255), 1)

    # draw anchors in img
    # get top anchor file
    file_path = os.path.join(pred_path, 'proposals_and_scores/val/'+cpkt_idx)
    anchor_file = os.path.join(file_path, sample_names + '.txt')
    proposals = np.loadtxt(anchor_file)
    # proposals = proposals[proposals[:, -1] > 0.05]
    proposals_anchors = proposals[:, :6]
    _, proposal_bev_norm_boxes = \
        anchor_projector.project_to_bev(proposals_anchors, area_extents[[0, 2]])
    proposal_bev_boxes = proposal_bev_norm_boxes * [800, 700, 800, 700]

    center_xy = np.zeros((proposal_bev_boxes.shape[0], 2), np.int32)
    center_xy[:, 0] = 1 / 2 * (proposal_bev_boxes[:, 0] + proposal_bev_boxes[:, 2])
    center_xy[:, 1] = 1 / 2 * (proposal_bev_boxes[:, 1] + proposal_bev_boxes[:, 3])
    # print(center_xy)

    # cluster
    db = DBSCAN(eps=5, min_samples=5).fit(center_xy)
    labels = db.labels_
    print(labels)

    cluster_num = max(labels) + 2
    all_color = np.random.randint(0,256,(cluster_num,3)).astype(np.uint32)

    for i in range(len(center_xy)):
        point = tuple(center_xy[i])
        color = all_color[labels[i]+1]
        cv2.circle(img, point, 1, (int(color[0]), int(color[1]), int(color[2])), 1)

    cv2.imwrite(first_name + '.png', img)

def main(_):
    default_pipeline_config_path = avod.root_dir() + \
        '/configs/pyramid_cars_with_aug_stack_5_tracking_pretrained.config'

    # Parse pipeline config
    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            default_pipeline_config_path,
            is_training=False)

    # Overwrite data split
    dataset_config.data_split = 'val'

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    dataset = build_dataset(eval_config, dataset_config)

    sample_name = '000132_000134'
    cpkt_idx = '7000'
    pred_path = model_config.paths_config.pred_dir
    draw_anchors(dataset, pred_path, cpkt_idx, sample_name)


if __name__ == '__main__':
    tf.app.run()






