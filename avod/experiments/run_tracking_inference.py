"""Detection model inferencing.

This runs the DetectionModel evaluator in test mode to output detections.
"""

import argparse
import os
import sys

import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.dt_avod_model import DtAvodModel
from avod.core.models.dt_rpn_model import DtRpnModel
from avod.core.dt_evaluator import DtEvaluator


def inference(model_config, eval_config,
              dataset_config, data_split,
              ckpt_indices):

    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)

    dataset_config.data_split = data_split
    dataset_config.data_split_dir = 'training'
    if data_split == 'test':
        dataset_config.data_split_dir = 'testing'

    eval_config.eval_mode = 'test'
    eval_config.evaluate_repeatedly = False

    dataset_config.has_labels = False
    # Enable this to see the actually memory being used
    eval_config.allow_gpu_mem_growth = True

    eval_config = config_builder.proto_to_obj(eval_config)
    # Grab the checkpoint indices to evaluate
    eval_config.ckpt_indices = ckpt_indices

    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config,
                                                 use_defaults=False)

    # Setup the model
    model_name = model_config.model_name
    # Overwrite repeated field
    model_config = config_builder.proto_to_obj(model_config)
    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]

    with tf.Graph().as_default():
        if model_name == 'dt_avod_model':
            model = DtAvodModel(model_config,
                              train_val_test=eval_config.eval_mode,
                              dataset=dataset)
        elif model_name == 'dt_rpn_model':
            model = DtRpnModel(model_config,
                             train_val_test=eval_config.eval_mode,
                             dataset=dataset)
        else:
            raise ValueError('Invalid model name {}'.format(model_name))

        model_evaluator = DtEvaluator(model, dataset_config, eval_config)
        model_evaluator.run_latest_checkpoints()


def main(_):
    parser = argparse.ArgumentParser()

    # Example usage
    checkpoint_name='pyramid_cars_with_aug_dt_5_tracking_corr_pretrained_new'
    data_split='test'
    ckpt_indices=[7]
    # Optional arg:

    parser.add_argument('--checkpoint_name',
                        type=str,
                        dest='checkpoint_name',
                        default=checkpoint_name,
                        help='Checkpoint name must be specified as a str\
                        and must match the experiment config file name.')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=data_split,
                        help='Data split must be specified e.g. val or test')

    parser.add_argument('--dataset',
                        type=str,
                        dest='dataset',
                        default='',
                        help='Dataset for training')

    parser.add_argument('--ckpt_indices',
                        type=int,
                        nargs='+',
                        dest='ckpt_indices',
                        default=ckpt_indices,
                        help='Checkpoint indices must be a set of \
                        integers with space in between -> 0 10 20 etc')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default='3',
                        help='CUDA device id')

    args = parser.parse_args()


    experiment_config = args.checkpoint_name + '.config'

    # Read the config from the experiment folder
    experiment_config_path = avod.root_dir() + '/data/outputs/' +\
        args.checkpoint_name + '/' + experiment_config

    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            experiment_config_path, args.dataset, is_training=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    inference(model_config, eval_config,
              dataset_config, args.data_split,
              args.ckpt_indices)


if __name__ == '__main__':
    tf.app.run()
