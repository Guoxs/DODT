"""Detection model evaluator.

This runs the DetectionModel evaluator.
"""

import argparse
import os

import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.dt_avod_model import DtAvodModel
from avod.core.models.dt_rpn_model import DtRpnModel
from avod.core.dt_evaluator import DtEvaluator


def evaluate(model_config, eval_config, dataset_config):

    # Parse eval config
    eval_mode = eval_config.eval_mode
    if eval_mode not in ['val', 'test']:
        raise ValueError('Evaluation mode can only be set to `val` or `test`')
    evaluate_repeatedly = eval_config.evaluate_repeatedly

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
    dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config,
                                                 use_defaults=False)

    # Setup the model
    model_name = model_config.model_name

    # Convert to object to overwrite repeated fields
    model_config = config_builder.proto_to_obj(model_config)

    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]

    with tf.Graph().as_default():
        if model_name == 'dt_avod_model':
            model = DtAvodModel(model_config, train_val_test=eval_mode,
                              dataset=dataset)
        elif model_name == 'dt_rpn_model':
            model = DtRpnModel(model_config, train_val_test=eval_mode,
                             dataset=dataset)
        else:
            raise ValueError('Invalid model name {}'.format(model_name))

        model_evaluator = DtEvaluator(model,
                                    dataset_config,
                                    eval_config)

        if evaluate_repeatedly:
            model_evaluator.repeated_checkpoint_run()
            # checkpoint_name = 'pyramid_cars_with_aug_dt_tracking_2'
            # root_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
            #             '/predictions/final_predictions_and_scores/val/1000/'
            # output_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
            #             '/predictions/kitti_tracking_native_eval/results/1000/data/'
            # model_evaluator.run_kitti_native_tracking_eval(root_dir, output_dir, 1000)
        else:
            model_evaluator.run_latest_checkpoints()


def main(_):
    parser = argparse.ArgumentParser()

    default_pipeline_config_path = avod.root_dir() + \
        '/configs/pyramid_cars_with_aug_dt_10_tracking_2.config'

    default_device = '3'

    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        default=default_pipeline_config_path,
                        help='Path to the pipeline config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default='val',
                        help='Data split for evaluation')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default=default_device,
                        help='CUDA device id')

    args = parser.parse_args()

    # Parse pipeline config
    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.pipeline_config_path,
            is_training=False)

    # Overwrite data split
    dataset_config.data_split = args.data_split

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    evaluate(model_config, eval_config, dataset_config)


if __name__ == '__main__':
    tf.app.run()
