import os
import numpy as np
import tensorflow as tf

import avod
from avod.core import trainer_utils
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.dt_avod_model import DtAvodModel
from avod.core.dt_inference_utils import get_avod_pred, \
                                convert_pred_to_kitti_format

from wavedata.tools.obj_detection.evaluation import three_d_iou


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def build_dataset(dataset_config):
    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)
    dataset_config.data_split = 'test'
    dataset_config.data_split_dir = 'testing'
    dataset_config.has_labels = False
    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []
     # Build the dataset object
    dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config,
                                                     use_defaults=False)
    return dataset

def model_setup(model_config):
    # Overwrite repeated field
    model_config = config_builder.proto_to_obj(model_config)
    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]
    model = DtAvodModel(model_config, train_val_test='test', dataset=dataset)
    return model

def create_sesson(model_config, cpkt_idx):
    checkpoint_dir = model_config.paths_config.checkpoint_dir

    saver = tf.train.Saver()
    trainer_utils.load_checkpoints(checkpoint_dir, saver)
    checkpoint_to_restore = saver.last_checkpoints[cpkt_idx]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver.restore(sess, checkpoint_to_restore)
    return sess

def get_kitti_pred(predictions, idx, dataset, box_rep):
    all_pred = get_avod_pred(predictions, box_rep)
    frame_mask_0 = np.where(all_pred[:, -1] == 0)[0]
    frame_mask_1 = np.where(all_pred[:, -1] == 1)[0]
    pred_frame_0 = all_pred[frame_mask_0]
    pred_frame_1 = all_pred[frame_mask_1]
    sample_name_0 = dataset.sample_names[idx][0]
    sample_name_1 = dataset.sample_names[idx][1]
    score_threshold = 0.6
    pred_frame_kitti_0 = convert_pred_to_kitti_format(
        pred_frame_0[:, :9], sample_name_0, dataset, score_threshold)
    pred_frame_kitti_1 = convert_pred_to_kitti_format(
        pred_frame_1[:, :9], sample_name_1, dataset, score_threshold)

    mask = pred_frame_0[:, 7] >= score_threshold
    corr_offset = pred_frame_0[mask][:, 9:12]
    return pred_frame_kitti_0, pred_frame_kitti_1, corr_offset

def cal_iou_mat(kitti_pred1, kitti_pred2, offsets):
    # [l, w, h, x, y, z, ry]
    box3d_1 = np.array(kitti_pred1[:, 8:16], dtype=np.float32)
    box3d_2 = np.array(kitti_pred2[:, 8:16], dtype=np.float32)
    # frame 1 add offsets
    box3d_1[:, 3:6] = box3d_1[:, 3:6]  + offsets
    # convert to [ry, l, h, w, tx, ty, tz]
    box3d_1 = box3d_1[:, [-2, 0, 2, 1, 3, 4, 5]]
    box3d_2 = box3d_2[:, [-2, 0, 2, 1, 3, 4, 5]]
    iou_3d = np.zeros((box3d_1.shape[0], box3d_2.shape[0]), dtype=np.float32)
    for i in range(len(box3d_1)):
        iou_3d[i] = three_d_iou(box3d_1[i], box3d_2)
    return iou_3d


checkpoint_name = 'pyramid_cars_with_aug_dt_tracking_stride_5'
# Read the config from the experiment folder
experiment_config_path = avod.root_dir() + '/data/outputs/' +\
        checkpoint_name + '/' + checkpoint_name + '.config'

model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            experiment_config_path, is_training=False)

dataset = build_dataset(dataset_config)
box_rep = model_config.avod_config.avod_box_representation

with tf.Graph().as_default():
    model = model_setup(model_config)
    prediction_dict = model.build()
    sess = create_sesson(model_config, cpkt_idx=107)
    for i in range(200):
        feed_dict = model.create_feed_dict(sample_index=i)
        predictions = sess.run(prediction_dict, feed_dict=feed_dict)
        pred_frame_kitti_0, pred_frame_kitti_1, corr_offset = \
            get_kitti_pred(predictions, i, dataset, box_rep)
        iou_3d = cal_iou_mat(pred_frame_kitti_0, pred_frame_kitti_1, corr_offset)
        print(iou_3d)

