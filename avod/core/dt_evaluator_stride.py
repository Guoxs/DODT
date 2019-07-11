"""Common functions for evaluating checkpoints.
"""
import collections
import time
import os
import numpy as np
from multiprocessing import Process

import tensorflow as tf

from avod.core import box_3d_encoder, box_8c_encoder, box_4c_encoder
from avod.core import evaluator_utils
from avod.core import summary_utils
from avod.core import trainer_utils
from avod.core.evaluator_utils import encoder_tracking_dets, \
    track_through_ious, convert_trajectory_to_kitti_format

from avod.core.models.dt_avod_model import DtAvodModel
from avod.core.models.dt_rpn_model import DtRpnModel

tf.logging.set_verbosity(tf.logging.INFO)

KEY_SUM_RPN_OBJ_LOSS = 'sum_rpn_obj_loss'
KEY_SUM_RPN_REG_LOSS = 'sum_rpn_reg_loss'
KEY_SUM_RPN_TOTAL_LOSS = 'sum_rpn_total_loss'
KEY_SUM_RPN_OBJ_ACC = 'sum_rpn_obj_accuracy'

KEY_SUM_AVOD_CLS_LOSS = 'sum_avod_cls_loss'
KEY_SUM_AVOD_REG_LOSS = 'sum_avod_reg_loss'
KEY_SUM_AVOD_CORR_LOSS = 'sum_avod_corr_loss'
KEY_SUM_AVOD_TOTAL_LOSS = 'sum_avod_total_loss'
KEY_SUM_AVOD_LOC_LOSS = 'sum_avod_loc_loss'
KEY_SUM_AVOD_ANG_LOSS = 'sum_avod_ang_loss'
KEY_SUM_AVOD_CLS_ACC = 'sum_avod_cls_accuracy'
KEY_NUM_VALID_REG_SAMPLES = 'num_valid_reg_samples'


class DtEvaluator:

    def __init__(self,
                 model,
                 dataset_config,
                 eval_config,
                 skip_evaluated_checkpoints=True,
                 eval_wait_interval=30,
                 do_kitti_native_eval=True,
                 do_kitti_native_tracking_eval=False):
        """Evaluator class for evaluating model's detection output.

        Args:
            model: An instance of DetectionModel
            dataset_config: Dataset protobuf configuration
            eval_config: Evaluation protobuf configuration
            skip_evaluated_checkpoints: (optional) Enables checking evaluation
                results directory and if the folder names with the checkpoint
                index exists, it 'assumes' that checkpoint has already been
                evaluated and skips that checkpoint.
            eval_wait_interval: (optional) The number of seconds between
                looking for a new checkpoint.
            do_kitti_native_eval: (optional) flag to enable running kitti native
                eval code.
        """

        # Get model configurations
        self.model = model
        self.dataset_config = dataset_config
        self.eval_config = eval_config

        self.model_config = model.model_config
        self.model_name = self.model_config.model_name
        self.full_model = isinstance(self.model, DtAvodModel)

        self.paths_config = self.model_config.paths_config
        self.checkpoint_dir = self.paths_config.checkpoint_dir

        self.skip_evaluated_checkpoints = skip_evaluated_checkpoints
        self.eval_wait_interval = eval_wait_interval

        self.do_kitti_native_eval = do_kitti_native_eval
        self.do_kitti_native_tracking_eval = do_kitti_native_tracking_eval

        # Create a variable tensor to hold the global step
        self.global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')

        eval_mode = eval_config.eval_mode
        if eval_mode not in ['val', 'test']:
            raise ValueError('Evaluation mode can only be set to `val`'
                             'or `test`')

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        if self.do_kitti_native_eval:
            if self.eval_config.eval_mode == 'val':
                # Copy kitti native eval code into the predictions folder
                evaluator_utils.copy_kitti_native_code(
                    self.model_config.checkpoint_name)

        if self.do_kitti_native_tracking_eval:
            # generate video frames for tracking evaluation
            self.video_frames = self.generate_video_frames()
            video_ids = self.video_frames.keys()
            # Copy kitti tracking native eval code into the predictions folder
            evaluator_utils.copy_kitti_native_tracking_code(
                self.model_config.checkpoint_name,
                video_ids, eval_mode)

        allow_gpu_mem_growth = self.eval_config.allow_gpu_mem_growth
        if allow_gpu_mem_growth:
            # GPU memory config
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = allow_gpu_mem_growth
            self._sess = tf.Session(config=config)
        else:
            self._sess = tf.Session()

        # The model should return a dictionary of predictions
        self._prediction_dict = self.model.build()
        if eval_mode == 'val':
            # Setup loss and summary writer in val mode only
            self._loss_dict, self._total_loss = \
                self.model.loss(self._prediction_dict)

            self.summary_writer, self.summary_merged = \
                evaluator_utils.set_up_summary_writer(
                    self.model_config, self._sess)

        else:
            self._loss_dict = None
            self._total_loss = None
            self.summary_writer = None
            self.summary_merged = None

        self._saver = tf.train.Saver()

        # Add maximum memory usage summary op
        # This op can only be run on device with gpu
        # so it's skipped on travis
        is_travis = 'TRAVIS' in os.environ
        if not is_travis:
            # tf 1.4
            # tf.summary.scalar('bytes_in_use',
            #                   tf.contrib.memory_stats.BytesInUse())
            tf.summary.scalar('max_bytes',
                              tf.contrib.memory_stats.MaxBytesInUse())

    def generate_video_frames(self):
        video_frames = {}
        dataset = self.model.dataset
        sample_names = dataset.sample_names
        for sample_name in sample_names:
            video_id = sample_name[0][:2]
            if not video_frames.__contains__(video_id):
                video_frames[video_id] = []
            txt_name = sample_name[0] + '_' + sample_name[1]
            video_frames[video_id].append(txt_name)

        video_frames = collections.OrderedDict(sorted(video_frames.items(),
                                                      key=lambda obj: obj[0]))
        return video_frames

    def run_checkpoint_once(self, checkpoint_to_restore):
        """Evaluates network metrics once over all the validation samples.

        Args:
            checkpoint_to_restore: The directory of the checkpoint to restore.
        """

        self._saver.restore(self._sess, checkpoint_to_restore)

        data_split = self.dataset_config.data_split
        predictions_base_dir = self.paths_config.pred_dir

        num_samples = self.model.dataset.num_samples
        train_val_test = self.model._train_val_test

        validation = train_val_test == 'val'

        global_step = trainer_utils.get_global_step(
            self._sess, self.global_step_tensor)

        # Rpn average losses dictionary
        if validation:
            eval_rpn_losses = self._create_rpn_losses_dict()

        # Add folders to save predictions
        prop_score_predictions_dir = predictions_base_dir + \
            "/proposals_and_scores/{}/{}".format(
                data_split, global_step)
        trainer_utils.create_dir(prop_score_predictions_dir)

        if self.full_model:
            # Make sure the box representation is valid
            box_rep = self.model_config.avod_config.avod_box_representation
            if box_rep not in ['box_3d', 'box_8c', 'box_8co',
                               'box_4c', 'box_4ca']:
                raise ValueError('Invalid box representation {}.'.
                                 format(box_rep))

            avod_predictions_dir = predictions_base_dir + \
                "/final_predictions_and_scores/{}/{}".format(
                    data_split, global_step)
            trainer_utils.create_dir(avod_predictions_dir)

            if box_rep in ['box_8c', 'box_8co', 'box_4c', 'box_4ca']:
                avod_box_corners_dir = predictions_base_dir + \
                    "/final_boxes_{}_and_scores/{}/{}".format(
                        box_rep, data_split, global_step)
                trainer_utils.create_dir(avod_box_corners_dir)

            # Avod average losses dictionary
            eval_avod_losses = self._create_avod_losses_dict()

        # Add folders to save predicitons for native kitti eval
        kitti_detection_eval_prediction_dir = predictions_base_dir + \
            "/kitti_detection_predictions_and_scores/{}/{}".format(
            data_split, global_step)
        trainer_utils.create_dir(kitti_detection_eval_prediction_dir)

        # Add folders to save predictions for native kitti tracking eval
        kitti_tracking_eval_prediction_dir = predictions_base_dir + \
            "/kitti_tracking_native_eval/results/{}/data/".format(global_step)
        trainer_utils.create_dir(kitti_tracking_eval_prediction_dir)

        num_valid_samples = 0

        # Keep track of feed_dict and inference time
        total_feed_dict_time = []
        total_inference_time = []

        # Run through a single epoch
        current_epoch = self.model.dataset.epochs_completed
        while current_epoch == self.model.dataset.epochs_completed:

            # Keep track of feed_dict speed
            start_time = time.time()
            feed_dict = self.model.create_feed_dict()
            feed_dict_time = time.time() - start_time

            # Get sample name from model
            sample_name = self.model.sample_info['sample_name']

            # File paths for saving proposals and predictions
            rpn_file_path = prop_score_predictions_dir + "/{}_{}.txt".format(
                sample_name[0], sample_name[1])

            if self.full_model:
                avod_file_path = avod_predictions_dir + \
                    "/{}_{}.txt".format(sample_name[0], sample_name[1])

                if box_rep in ['box_8c', 'box_8co', 'box_4c', 'box_4ca']:
                    avod_box_corners_file_path = avod_box_corners_dir + \
                        '/{}_{}.txt'.format(sample_name[0], sample_name[1])

            # kitti detection native eval, only eval first frame
            kitti_eval_file_path = kitti_detection_eval_prediction_dir + \
                                   '/{}.txt'.format(sample_name[0])

            num_valid_samples += 1
            print("Step {}: {} / {}, Inference on sample {}_{}".format(
                global_step, num_valid_samples, num_samples,
                sample_name[0], sample_name[1]))

            # Do predictions, loss calculations, and summaries
            if validation:
                if self.summary_merged is not None:
                    predictions, eval_losses, eval_total_loss, summary_out = \
                        self._sess.run([self._prediction_dict,
                                        self._loss_dict,
                                        self._total_loss,
                                        self.summary_merged],
                                       feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_out, global_step)

                else:
                    predictions, eval_losses, eval_total_loss = \
                        self._sess.run([self._prediction_dict,
                                        self._loss_dict,
                                        self._total_loss],
                                       feed_dict=feed_dict)

                rpn_objectness_loss = eval_losses[DtRpnModel.LOSS_RPN_OBJECTNESS]
                rpn_regression_loss = eval_losses[DtRpnModel.LOSS_RPN_REGRESSION]

                self._update_rpn_losses(eval_rpn_losses,
                                        rpn_objectness_loss,
                                        rpn_regression_loss,
                                        eval_total_loss,
                                        global_step)

                # Save proposals
                proposals_and_scores = \
                    self.get_rpn_proposals_and_scores(predictions)
                np.savetxt(rpn_file_path, proposals_and_scores, fmt='%.3f')

                # Save predictions
                predictions_and_scores = \
                    self.get_avod_predicted_boxes_3d_and_scores(predictions,
                                                                box_rep)
                np.savetxt(avod_file_path, predictions_and_scores, fmt='%.5f')

                # Save predictions for kitti native deteciton eval
                first_frame_indices = np.where(predictions_and_scores[:, -1] == 0)
                kitti_eval_prediction_and_scores = \
                    predictions_and_scores[first_frame_indices][:, :9]
                np.savetxt(kitti_eval_file_path, kitti_eval_prediction_and_scores, fmt='%.5f')

                if self.full_model:
                    if box_rep in ['box_3d', 'box_4ca']:
                        self._update_avod_box_cls_loc_orient_losses(
                            eval_avod_losses,
                            eval_losses,
                            eval_total_loss,
                            global_step)

                    elif box_rep in ['box_8c', 'box_8co', 'box_4c']:
                        self._update_avod_box_cls_loc_losses(
                            eval_avod_losses,
                            eval_losses,
                            eval_total_loss,
                            global_step)

                    if box_rep != 'box_3d':
                        # Save box corners for all box reps
                        # except for box_3d which is not a corner rep
                        predicted_box_corners_and_scores = \
                            self.get_avod_predicted_box_corners_and_scores(
                                predictions, box_rep)
                        np.savetxt(avod_box_corners_file_path,
                                   predicted_box_corners_and_scores,
                                   fmt='%.5f')

                # Calculate accuracies
                self.get_cls_accuracy(predictions,
                                      eval_avod_losses,
                                      eval_rpn_losses,
                                      global_step)
                print("Step {}: Total time {} s".format(
                    global_step, time.time() - start_time))

            else:
                # Test mode --> train_val_test == 'test'
                inference_start_time = time.time()
                # Don't calculate loss or run summaries for test
                predictions = self._sess.run(self._prediction_dict,
                                             feed_dict=feed_dict)
                inference_time = time.time() - inference_start_time

                # Add times to list
                total_feed_dict_time.append(feed_dict_time)
                total_inference_time.append(inference_time)

                proposals_and_scores = \
                    self.get_rpn_proposals_and_scores(predictions)
                predictions_and_scores = \
                    self.get_avod_predicted_boxes_3d_and_scores(predictions,
                                                                box_rep)

                np.savetxt(rpn_file_path, proposals_and_scores, fmt='%.3f')
                np.savetxt(avod_file_path, predictions_and_scores, fmt='%.5f')

        # end while current_epoch == model.dataset.epochs_completed:

        if validation:
            self.save_proposal_losses_results(eval_rpn_losses,
                                              num_valid_samples,
                                              global_step,
                                              predictions_base_dir)
            if self.full_model:
                self.save_prediction_losses_results(
                    eval_avod_losses,
                    num_valid_samples,
                    global_step,
                    predictions_base_dir,
                    box_rep=box_rep)

                # Kitti native evaluation, do this during validation
                # and when running dtAvod model.
                # Store predictions in kitti format
                if self.do_kitti_native_eval:
                    self.run_kitti_native_eval(global_step)
                if self.do_kitti_native_tracking_eval:
                    self.run_kitti_native_tracking_eval(
                        avod_predictions_dir,
                        kitti_tracking_eval_prediction_dir,
                        global_step
                    )

        else:
            # Test mode --> train_val_test == 'test'
            evaluator_utils.print_inference_time_statistics(
                total_feed_dict_time, total_inference_time)

        print("Step {}: Finished evaluation, results saved to {}".format(
            global_step, prop_score_predictions_dir))

    def run_latest_checkpoints(self):
        """Evaluation function for evaluating all the existing checkpoints.
        This function just runs through all the existing checkpoints.

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        # Load the latest checkpoints available
        trainer_utils.load_checkpoints(self.checkpoint_dir,
                                       self._saver)

        num_checkpoints = len(self._saver.last_checkpoints)

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts(
                self.model_config, self.model_name)

        ckpt_indices = np.asarray(self.eval_config.ckpt_indices)
        if ckpt_indices is not None:
            if ckpt_indices[0] == -1:
                # Restore the most recent checkpoint
                ckpt_idx = num_checkpoints - 1
                ckpt_indices = [ckpt_idx]
            for ckpt_idx in ckpt_indices:
                checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                self.run_checkpoint_once(checkpoint_to_restore)

        else:
            last_checkpoint_id = -1
            number_of_evaluations = 0
            # go through all existing checkpoints
            for ckpt_idx in range(num_checkpoints):
                checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                ckpt_id = evaluator_utils.strip_checkpoint_id(
                    checkpoint_to_restore)

                # Check if checkpoint has been evaluated already
                already_evaluated = ckpt_id in already_evaluated_ckpts
                if already_evaluated or ckpt_id <= last_checkpoint_id:
                    number_of_evaluations = max((ckpt_idx + 1,
                                                 number_of_evaluations))
                    continue

                self.run_checkpoint_once(checkpoint_to_restore)
                number_of_evaluations += 1

                # Save the id of the latest evaluated checkpoint
                last_checkpoint_id = ckpt_id

    def repeated_checkpoint_run(self):
        """Periodically evaluates the checkpoints inside the `checkpoint_dir`.

        This function evaluates all the existing checkpoints as they are being
        generated. If there are none, it sleeps until new checkpoints become
        available. Since there is no synchronization guarantee for the trainer
        and evaluator, at each iteration it reloads all the checkpoints and
        searches for the last checkpoint to continue from. This is meant to be
        called in parallel to the trainer to evaluate the models regularly.

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        # Copy kitti native eval code into the predictions folder
        if self.do_kitti_native_eval:
            evaluator_utils.copy_kitti_native_code(
                self.model_config.checkpoint_name)

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts(
                self.model_config, self.full_model)
        tf.logging.info(
            'Starting evaluation at ' +
            time.strftime(
                '%Y-%m-%d-%H:%M:%S',
                time.gmtime()))

        last_checkpoint_id = -1
        number_of_evaluations = 0
        while True:
            # Load current checkpoints available
            trainer_utils.load_checkpoints(self.checkpoint_dir,
                                           self._saver)
            num_checkpoints = len(self._saver.last_checkpoints)

            start = time.time()

            if number_of_evaluations >= num_checkpoints:
                tf.logging.info('No new checkpoints found in %s.'
                                'Will try again in %d seconds',
                                self.checkpoint_dir,
                                self.eval_wait_interval)
            else:
                for ckpt_idx in range(num_checkpoints):
                    checkpoint_to_restore = \
                        self._saver.last_checkpoints[ckpt_idx]
                    ckpt_id = evaluator_utils.strip_checkpoint_id(
                        checkpoint_to_restore)

                    # Check if checkpoint has been evaluated already
                    already_evaluated = ckpt_id in already_evaluated_ckpts
                    if already_evaluated or ckpt_id <= last_checkpoint_id:
                        number_of_evaluations = max((ckpt_idx + 1,
                                                     number_of_evaluations))
                        continue

                    self.run_checkpoint_once(checkpoint_to_restore)
                    number_of_evaluations += 1

                    # Save the id of the latest evaluated checkpoint
                    last_checkpoint_id = ckpt_id

            time_to_next_eval = start + self.eval_wait_interval - time.time()
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)

    def _update_rpn_losses(self,
                           eval_rpn_losses,
                           rpn_objectness_loss,
                           rpn_regression_loss,
                           eval_total_loss,
                           global_step):
        """Helper function to calculate the evaluation average losses.

        Args:
            eval_rpn_losses: A dictionary containing all the average
                losses.
            rpn_objectness_loss: A list of two scalar losses of rpn objectness.
            rpn_regression_loss: A list of two scalar losses of rpn objectness.
            eval_total_loss: A scalar loss of rpn total loss.
            global_step: Global step at which the metrics are computed.
        """

        if self.full_model:
            # The full model total_loss will be the sum of Rpn and Avod
            # so calculate the total rpn loss instead
            rpn_total_loss = rpn_objectness_loss + rpn_regression_loss
        else:
            rpn_total_loss = eval_total_loss

        print("Step {}: Eval RPN Loss: objectness {:.3f}, "
              "regression {:.3f}, total {:.3f}".format(
                    global_step,
                    rpn_objectness_loss,
                    rpn_regression_loss,
                    rpn_total_loss))

        # Get the loss sums from the losses dict
        sum_rpn_obj_loss = eval_rpn_losses[KEY_SUM_RPN_OBJ_LOSS]
        sum_rpn_reg_loss = eval_rpn_losses[KEY_SUM_RPN_REG_LOSS]
        sum_rpn_total_loss = eval_rpn_losses[KEY_SUM_RPN_TOTAL_LOSS]


        sum_rpn_obj_loss += rpn_objectness_loss
        sum_rpn_reg_loss += rpn_regression_loss
        sum_rpn_total_loss += rpn_total_loss

        # update the losses sums
        eval_rpn_losses.update({KEY_SUM_RPN_OBJ_LOSS:
                                sum_rpn_obj_loss})

        eval_rpn_losses.update({KEY_SUM_RPN_REG_LOSS:
                                sum_rpn_reg_loss})

        eval_rpn_losses.update({KEY_SUM_RPN_TOTAL_LOSS:
                                sum_rpn_total_loss})

    def _update_avod_box_cls_loc_orient_losses(self,
                                               eval_avod_losses,
                                               eval_losses,
                                               eval_total_loss,
                                               global_step):
        """Helper function to calculate the evaluation average losses.

        Note: This function evaluates classification, regression/offsets
            and orientation losses.

        Args:
            eval_avod_losses: A dictionary containing all the average
                losses.
            eval_losses: A dictionary containing the current evaluation
                losses.
            eval_total_loss: A scalar loss of model total loss.
            global_step: Global step at which the metrics are computed.
        """
        # Get the loss sums from the losses dict
        sum_avod_cls_loss = eval_avod_losses[KEY_SUM_AVOD_CLS_LOSS]
        sum_avod_reg_loss = eval_avod_losses[KEY_SUM_AVOD_REG_LOSS]
        sum_avod_corr_loss = eval_avod_losses[KEY_SUM_AVOD_CORR_LOSS]
        sum_avod_total_loss = eval_avod_losses[KEY_SUM_AVOD_TOTAL_LOSS]

        # for the full model, we expect a total of 4 losses
        assert (len(eval_losses) > 2)
        avod_classification_loss = \
            eval_losses[DtAvodModel.LOSS_FINAL_CLASSIFICATION]
        avod_regression_loss = \
            eval_losses[DtAvodModel.LOSS_FINAL_REGRESSION]
        avod_correlation_loss = \
            eval_losses[DtAvodModel.LOSS_FINAL_CORRELATION]

        avod_localization_loss = \
            eval_losses[DtAvodModel.LOSS_FINAL_LOCALIZATION]
        avod_orientation_loss = \
            eval_losses[DtAvodModel.LOSS_FINAL_ORIENTATION]

        sum_avod_cls_loss += avod_classification_loss
        sum_avod_reg_loss += avod_regression_loss
        sum_avod_corr_loss += avod_correlation_loss
        sum_avod_total_loss += eval_total_loss

        # update the losses sums
        eval_avod_losses.update({KEY_SUM_AVOD_CLS_LOSS:
                                 sum_avod_cls_loss})

        eval_avod_losses.update({KEY_SUM_AVOD_REG_LOSS:
                                 sum_avod_reg_loss})

        eval_avod_losses.update({KEY_SUM_AVOD_CORR_LOSS:
                                 sum_avod_corr_loss})

        eval_avod_losses.update({KEY_SUM_AVOD_TOTAL_LOSS:
                                 sum_avod_total_loss})

        # Only add localization and orientation losses if valid
        # (greater than 0)
        if (avod_localization_loss > 0.0) and \
                (avod_orientation_loss > 0.0):

            sum_avod_loc_loss = eval_avod_losses[KEY_SUM_AVOD_LOC_LOSS]
            sum_avod_ang_loss = eval_avod_losses[KEY_SUM_AVOD_ANG_LOSS]

            sum_avod_loc_loss += avod_localization_loss
            sum_avod_ang_loss += avod_orientation_loss

            eval_avod_losses.update({KEY_SUM_AVOD_LOC_LOSS:
                                     sum_avod_loc_loss})
            eval_avod_losses.update({KEY_SUM_AVOD_ANG_LOSS:
                                     sum_avod_ang_loss})
            num_valid_regression_samples = \
                eval_avod_losses[KEY_NUM_VALID_REG_SAMPLES]
            num_valid_regression_samples += 1
            eval_avod_losses.update({KEY_NUM_VALID_REG_SAMPLES:
                                     num_valid_regression_samples})

        print("Step {}: Eval AVOD Loss: "
              "classification {:.3f}, "
              "regression {:.3f}, "
              "correlation {: .3f},"
              "total {:.3f}".format(
                global_step,
                avod_classification_loss,
                avod_regression_loss,
                avod_correlation_loss,
                eval_total_loss))

        print("Step {}: Eval AVOD Loss: "
              "localization {:.3f}, "
              "orientation {:.3f}".format(
                global_step,
                avod_localization_loss,
                avod_orientation_loss))

    def _update_avod_box_cls_loc_losses(self,
                                        eval_avod_losses,
                                        eval_losses,
                                        eval_total_loss,
                                        global_step):
        """Helper function to calculate the evaluation average losses.

        Note: This function evaluates only classification and regression/offsets
            losses.

        Args:
            eval_avod_losses: A dictionary containing all the average
                losses.
            eval_losses: A dictionary containing the current evaluation
                losses.
            eval_total_loss: A scalar loss of model total loss.
            global_step: Global step at which the metrics are computed.
        """

        sum_avod_cls_loss = eval_avod_losses[KEY_SUM_AVOD_CLS_LOSS]
        sum_avod_reg_loss = eval_avod_losses[KEY_SUM_AVOD_REG_LOSS]
        sum_avod_corr_loss = eval_avod_losses[KEY_SUM_AVOD_CORR_LOSS]
        sum_avod_total_loss = eval_avod_losses[KEY_SUM_AVOD_TOTAL_LOSS]

        # for the full model, we expect a total of 4 losses
        assert (len(eval_losses) > 2)
        avod_classification_loss = \
            eval_losses[DtAvodModel.LOSS_FINAL_CLASSIFICATION]
        avod_regression_loss = \
            eval_losses[DtAvodModel.LOSS_FINAL_REGRESSION]
        avod_correlation_loss = \
            eval_losses[DtAvodModel.LOSS_FINAL_CORRELATION]

        avod_localization_loss = \
            eval_losses[DtAvodModel.LOSS_FINAL_LOCALIZATION]

        sum_avod_cls_loss += avod_classification_loss
        sum_avod_reg_loss += avod_regression_loss
        sum_avod_corr_loss += avod_correlation_loss
        sum_avod_total_loss += eval_total_loss

        eval_avod_losses.update({KEY_SUM_AVOD_CLS_LOSS:
                                 sum_avod_cls_loss})

        eval_avod_losses.update({KEY_SUM_AVOD_REG_LOSS:
                                 sum_avod_reg_loss})

        eval_avod_losses.update({KEY_SUM_AVOD_CORR_LOSS:
                                     sum_avod_corr_loss})

        eval_avod_losses.update({KEY_SUM_AVOD_TOTAL_LOSS:
                                 sum_avod_total_loss})

        # Only add localization and orientation losses if valid
        # (greater than 0)
        if (avod_localization_loss > 0.0):

            sum_avod_loc_loss = eval_avod_losses[KEY_SUM_AVOD_LOC_LOSS]

            sum_avod_loc_loss += avod_localization_loss

            eval_avod_losses.update({KEY_SUM_AVOD_LOC_LOSS:
                                     sum_avod_loc_loss})
            num_valid_regression_samples = \
                eval_avod_losses[KEY_NUM_VALID_REG_SAMPLES]
            num_valid_regression_samples += 1
            eval_avod_losses.update({KEY_NUM_VALID_REG_SAMPLES:
                                     num_valid_regression_samples})

        print("Step {}: Eval AVOD Loss: "
              "classification {:.3f}, "
              "regression {:.3f}, "
              "correlation {: .3f},"
              "total {:.3f}".format(
                global_step,
                avod_classification_loss,
                avod_regression_loss,
                avod_correlation_loss,
                eval_total_loss))

        print("Step {}: Eval AVOD Loss: "
              "localization {:.3f}, ".format(
                global_step,
                avod_localization_loss))

    def save_proposal_losses_results(self,
                                     eval_rpn_losses,
                                     num_valid_samples,
                                     global_step,
                                     predictions_base_dir):
        """Helper function to save the RPN loss evaluation results.
        """
        sum_rpn_obj_loss = eval_rpn_losses[KEY_SUM_RPN_OBJ_LOSS]
        sum_rpn_reg_loss = eval_rpn_losses[KEY_SUM_RPN_REG_LOSS]
        sum_rpn_total_loss = eval_rpn_losses[KEY_SUM_RPN_TOTAL_LOSS]
        sum_rpn_obj_accuracy = eval_rpn_losses[KEY_SUM_RPN_OBJ_ACC]

        # Calculate average loss and accuracy
        avg_rpn_obj_loss = sum_rpn_obj_loss / num_valid_samples
        avg_rpn_reg_loss = sum_rpn_reg_loss / num_valid_samples
        avg_rpn_total_loss = sum_rpn_total_loss / num_valid_samples
        avg_rpn_obj_accuracy = sum_rpn_obj_accuracy / num_valid_samples

        print("Step {}: Average RPN Losses: objectness {:.3f}, "
              "regression {:.3f}, total {:.3f}".format(global_step,
                                                       avg_rpn_obj_loss,
                                                       avg_rpn_reg_loss,
                                                       avg_rpn_total_loss))
        print("Step {}: Average Objectness Accuracy:{} ".format(
            global_step,
            avg_rpn_obj_accuracy))

        # Append to end of file
        avg_loss_file_path = predictions_base_dir + '/rpn_avg_losses.csv'
        with open(avg_loss_file_path, 'ba') as fp:
            np.savetxt(fp,
                       np.reshape([global_step,
                                   avg_rpn_obj_loss,
                                   avg_rpn_reg_loss,
                                   avg_rpn_total_loss],
                                  (1, 4)),
                       fmt='%d, %.5f, %.5f, %.5f')

        avg_acc_file_path = predictions_base_dir + '/rpn_avg_obj_acc.csv'
        with open(avg_acc_file_path, 'ba') as fp:
            np.savetxt(
                fp, np.reshape(
                    [global_step, avg_rpn_obj_accuracy],
                    (1, 2)),
                fmt='%d, %.5f')

    def save_prediction_losses_results(self,
                                       eval_avod_losses,
                                       num_valid_samples,
                                       global_step,
                                       predictions_base_dir,
                                       box_rep):
        """Helper function to save the AVOD loss evaluation results.

        Args:
            eval_avod_losses: A dictionary containing the loss sums
            num_valid_samples: An int, number of valid evaluated samples
                i.e. samples with valid ground-truth.
            global_step: Global step at which the metrics are computed.
            predictions_base_dir: Base directory for storing the results.
            box_rep: A string, the format of the 3D bounding box
                one of 'box_3d', 'box_8c' etc.
        """
        sum_avod_cls_loss = eval_avod_losses[KEY_SUM_AVOD_CLS_LOSS]
        sum_avod_reg_loss = eval_avod_losses[KEY_SUM_AVOD_REG_LOSS]
        sum_avod_corr_loss = eval_avod_losses[KEY_SUM_AVOD_CORR_LOSS]
        sum_avod_total_loss = eval_avod_losses[KEY_SUM_AVOD_TOTAL_LOSS]

        sum_avod_loc_loss = eval_avod_losses[KEY_SUM_AVOD_LOC_LOSS]
        sum_avod_ang_loss = eval_avod_losses[KEY_SUM_AVOD_ANG_LOSS]

        sum_avod_cls_accuracy = eval_avod_losses[KEY_SUM_AVOD_CLS_ACC]

        num_valid_regression_samples = eval_avod_losses[KEY_NUM_VALID_REG_SAMPLES]

        avg_avod_cls_loss = sum_avod_cls_loss / num_valid_samples
        avg_avod_reg_loss = sum_avod_reg_loss / num_valid_samples
        avg_avod_corr_loss = sum_avod_corr_loss / num_valid_samples
        avg_avod_total_loss = sum_avod_total_loss / num_valid_samples

        if num_valid_regression_samples > 0:
            avg_avod_loc_loss = \
                sum_avod_loc_loss / num_valid_regression_samples

            if box_rep in ['box_3d', 'box_4ca']:
                avg_avod_ang_loss = \
                    sum_avod_ang_loss / num_valid_regression_samples
        else:
            avg_avod_loc_loss = 0
            avg_avod_ang_loss = 0

        avg_avod_cls_accuracy = sum_avod_cls_accuracy / num_valid_samples

        # Write summaries
        summary_utils.add_scalar_summary(
            'avod_losses/classification/classification',
            avg_avod_cls_loss,
            self.summary_writer, global_step)
        summary_utils.add_scalar_summary(
            'avod_losses/regression/regression_total',
            avg_avod_reg_loss,
            self.summary_writer, global_step)
        summary_utils.add_scalar_summary(
            'avod_losses/correlation/correlation_total',
            avg_avod_corr_loss,
            self.summary_writer, global_step)

        summary_utils.add_scalar_summary(
            'avod_losses/regression/localization',
            avg_avod_loc_loss,
            self.summary_writer, global_step)
        if box_rep in ['box_3d', 'box_4ca']:
            summary_utils.add_scalar_summary(
                'avod_losses/regression/orientation',
                avg_avod_ang_loss,
                self.summary_writer, global_step)

        print("Step {}: Average AVOD Losses: "
              "cls {:.5f}, "
              "reg {:.5f}, "
              "corr {: .5},"
              "total {:.5f} ".format(
                global_step,
                avg_avod_cls_loss,
                avg_avod_reg_loss,
                avg_avod_corr_loss,
                avg_avod_total_loss,
                  ))

        if box_rep in ['box_3d', 'box_4ca']:
            print("Step {} Average AVOD Losses: "
                  "loc {:.5f} "
                  "ang {:.5f}".format(
                    global_step,
                    avg_avod_loc_loss,
                    avg_avod_ang_loss,
                      ))
        elif box_rep in ['box_8c', 'box_8co', 'box_4c']:
            print("Step {} Average AVOD Losses: "
                  "loc {:.5f} ".format(
                    global_step,
                    avg_avod_loc_loss
                      ))
        else:
            raise NotImplementedError('Print average loss not implemented')

        print("Step {}: Average Classification Accuracy: {} ".format(
            global_step, avg_avod_cls_accuracy))

        # Append to end of file
        avg_loss_file_path = predictions_base_dir + '/avod_avg_losses.csv'
        if box_rep in ['box_3d', 'box_4ca']:
            with open(avg_loss_file_path, 'ba') as fp:
                np.savetxt(fp,
                           [np.hstack(
                            [global_step,
                                avg_avod_cls_loss,
                                avg_avod_reg_loss,
                                avg_avod_corr_loss,
                                avg_avod_total_loss,

                                avg_avod_loc_loss,
                                avg_avod_ang_loss,
                             ]
                            )],
                           fmt='%d, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f')
        elif box_rep in ['box_8c', 'box_8co', 'box_4c']:
            with open(avg_loss_file_path, 'ba') as fp:
                np.savetxt(fp,
                           [np.hstack(
                            [global_step,
                                avg_avod_cls_loss,
                                avg_avod_reg_loss,
                                avg_avod_corr_loss,
                                avg_avod_total_loss,

                                avg_avod_loc_loss,
                             ]
                            )],
                           fmt='%d, %.5f, %.5f, %.5f, %.5f, %.5f')
        else:
            raise NotImplementedError('Saving losses not implemented')

        avg_acc_file_path = predictions_base_dir + '/avod_avg_cls_acc.csv'
        with open(avg_acc_file_path, 'ba') as fp:
            np.savetxt(
                fp, np.reshape(
                    [global_step, avg_avod_cls_accuracy],
                    (1, 2)),
                fmt='%d, %.5f')

    def _create_avod_losses_dict(self):
        """Returns a dictionary of the losses sum for averaging.
        """
        eval_avod_losses = dict()
        # Initialize Avod average losses
        eval_avod_losses[KEY_SUM_AVOD_CLS_LOSS] = 0
        eval_avod_losses[KEY_SUM_AVOD_REG_LOSS] = 0
        eval_avod_losses[KEY_SUM_AVOD_CORR_LOSS] = 0
        eval_avod_losses[KEY_SUM_AVOD_TOTAL_LOSS] = 0

        eval_avod_losses[KEY_SUM_AVOD_LOC_LOSS] = 0
        eval_avod_losses[KEY_SUM_AVOD_ANG_LOSS] = 0

        eval_avod_losses[KEY_SUM_AVOD_CLS_ACC] = 0

        # Number of samples that got regressed because
        # they were classified correctly
        eval_avod_losses[KEY_NUM_VALID_REG_SAMPLES] = 0

        return eval_avod_losses

    def _create_rpn_losses_dict(self):
        """Returns a dictionary of the losses sum for averaging.
        """
        eval_rpn_losses = dict()

        # Initialize Rpn average losses
        eval_rpn_losses[KEY_SUM_RPN_OBJ_LOSS] = 0
        eval_rpn_losses[KEY_SUM_RPN_REG_LOSS] = 0
        eval_rpn_losses[KEY_SUM_RPN_TOTAL_LOSS] = 0

        eval_rpn_losses[KEY_SUM_RPN_OBJ_ACC] = 0

        return eval_rpn_losses

    def get_evaluated_ckpts(self,
                            model_config,
                            model_name):
        """Finds the evaluated checkpoints.

        Examines the evaluation average losses file to find the already
        evaluated checkpoints.

        Args:
            model_config: Model protobuf configuration
            model_name: A string representing the model name.

        Returns:
            already_evaluated_ckpts: A list of checkpoint indices, or an
                empty list if no evaluated indices are found.
        """

        already_evaluated_ckpts = []

        # check for previously evaluated checkpoints
        # regardless of model, we are always evaluating rpn, but we do
        # this check based on model in case the evaluator got interrupted
        # and only saved results for one model
        paths_config = model_config.paths_config

        predictions_base_dir = paths_config.pred_dir
        if model_name == 'avod_model':
            avg_loss_file_path = predictions_base_dir + '/avod_avg_losses.csv'
        else:
            avg_loss_file_path = predictions_base_dir + '/rpn_avg_losses.csv'

        if os.path.exists(avg_loss_file_path):
            avg_losses = np.loadtxt(avg_loss_file_path, delimiter=',')
            if avg_losses.ndim == 1:
                # one entry
                already_evaluated_ckpts = np.asarray(
                    [avg_losses[0]], np.int32)
            else:
                already_evaluated_ckpts = np.asarray(avg_losses[:, 0],
                                                     np.int32)

        return already_evaluated_ckpts

    def get_cls_accuracy(self,
                         predictions,
                         eval_avod_losses,
                         eval_rpn_losses,
                         global_step):
        """Updates the calculated accuracies for rpn and avod losses.

        Args:
            predictions: A dictionary containing the model outputs.
            eval_avod_losses: A dictionary containing all the avod averaged
                losses.
            eval_rpn_losses: A dictionary containing all the rpn averaged
                losses.
            global_step: Current global step that is being evaluated.
        """

        objectness_pred = predictions[DtRpnModel.PRED_MB_OBJECTNESS]
        objectness_gt = predictions[DtRpnModel.PRED_MB_OBJECTNESS_GT]
        objectness_accuracy = [self.calculate_cls_accuracy(pred, gt)
                for pred, gt in zip(objectness_pred, objectness_gt)]
        objectness_accuracy = np.mean(objectness_accuracy)

        # get this from the key
        sum_rpn_obj_accuracy = eval_rpn_losses[KEY_SUM_RPN_OBJ_ACC]
        sum_rpn_obj_accuracy += objectness_accuracy
        eval_rpn_losses.update({KEY_SUM_RPN_OBJ_ACC:
                                sum_rpn_obj_accuracy})
        print("Step {}: RPN Objectness Accuracy: {}".format(
            global_step, objectness_accuracy))

        if self.full_model:
            classification_pred = \
                predictions[DtAvodModel.PRED_MB_CLASSIFICATION_SOFTMAX]
            classification_gt = \
                predictions[DtAvodModel.PRED_MB_CLASSIFICATIONS_GT]
            classification_accuracy = [self.calculate_cls_accuracy(pred, gt)
                for pred, gt in zip(classification_pred, classification_gt)]
            classification_accuracy = np.mean(classification_accuracy)

            sum_avod_cls_accuracy = eval_avod_losses[KEY_SUM_AVOD_CLS_ACC]
            sum_avod_cls_accuracy += classification_accuracy
            eval_avod_losses.update({KEY_SUM_AVOD_CLS_ACC:
                                     sum_avod_cls_accuracy})

            print("Step {}: AVOD Classification Accuracy: {}".format(
                global_step, classification_accuracy))

    def calculate_cls_accuracy(self, cls_pred, cls_gt):
        """Calculates accuracy of predicted objectness/classification wrt to
        the labels

        Args:
            cls_pred: A numpy array containing the predicted
            objectness/classification values in the form (mini_batches, 2)
            cls_gt: A numpy array containing the ground truth
            objectness/classification values in the form (mini_batches, 2)

        Returns:
            accuracy: A scalar value representing the accuracy
        """
        correct_prediction = np.equal(np.argmax(cls_pred, 1),
                                      np.argmax(cls_gt, 1))
        accuracy = np.mean(correct_prediction)
        return accuracy

    def get_rpn_proposals_and_scores(self, predictions):
        """Returns the proposals and scores stacked for saving to file.

        Args:
            predictions: A dictionary containing the model outputs.

        Returns:
            proposals_and_scores: A list of two numpy array of shape
            (number_of_proposals, 8), containing the rpn proposal boxes and scores.
            [x, y, z, w, h, l, r, frame_mark]
        """

        top_anchors = predictions[DtRpnModel.PRED_TOP_ANCHORS]
        softmax_scores = predictions[DtRpnModel.PRED_TOP_OBJECTNESS_SOFTMAX]

        top_proposals = [box_3d_encoder.anchors_to_box_3d(anchor)
                         for anchor in top_anchors]

        proposals_and_scores = [np.column_stack((proposals,scores))
                    for proposals, scores in zip(top_proposals, softmax_scores)]

        zero_mark = np.zeros((proposals_and_scores[0].shape[0], 1))
        one_mark = np.ones((proposals_and_scores[1].shape[0], 1))

        # add frame id in the end
        proposals_and_scores[0] = np.column_stack((
            proposals_and_scores[0], zero_mark))

        proposals_and_scores[1] = np.column_stack((
            proposals_and_scores[1], one_mark))

        # concatenate two frame proposals
        proposals_and_scores = np.concatenate(proposals_and_scores, axis=0)

        return proposals_and_scores

    def get_avod_predicted_boxes_3d_and_scores(self, predictions, box_rep):
        """Returns the predictions and scores stacked for saving to file.

        Args:
            predictions: A dictionary containing the model outputs.
            box_rep: A string indicating the format of the 3D bounding
                boxes i.e. 'box_3d', 'box_8c' etc

        Returns:
            predictions_and_scores: A numpy array of shape
                (number_of_predicted_boxes, 13), containing the final prediction
                boxes, orientations, scores, and types, frame no.
                [x, y, z, w, h, l, r, score, type, delta_x, delta_y, delta_z, delta_h, frame_mark]
        """
        if box_rep == 'box_3d':
            # Convert anchors + orientation to box_3d
            final_pred_anchors = predictions[DtAvodModel.PRED_TOP_PREDICTION_ANCHORS]
            final_pred_orientations = predictions[DtAvodModel.PRED_TOP_ORIENTATIONS]

            size = len(final_pred_anchors)

            final_pred_boxes_3d = [None] * size
            for i in range(size):
                final_pred_boxes_3d[i] = box_3d_encoder.anchors_to_box_3d(
                    final_pred_anchors[i], fix_lw=True)
                final_pred_boxes_3d[i][:, 6] = final_pred_orientations[i]

        elif box_rep in ['box_8c', 'box_8co', 'box_4c']:
            # Predictions are in box_3d format already
            final_pred_boxes_3d = predictions[DtAvodModel.PRED_TOP_PREDICTION_BOXES_3D]

        elif box_rep == 'box_4ca':
            # boxes_3d from boxes_4c
            final_pred_boxes_3d = predictions[DtAvodModel.PRED_TOP_PREDICTION_BOXES_3D]
            # Predicted orientation from layers
            final_pred_orientations = predictions[DtAvodModel.PRED_TOP_ORIENTATIONS]

            size = len(final_pred_boxes_3d)

            for i in range(size):
                # Calculate difference between box_3d and predicted angle
                ang_diff = final_pred_boxes_3d[i][:, 6] - final_pred_orientations[i]

                # Wrap differences between -pi and pi
                two_pi = 2 * np.pi
                ang_diff[ang_diff < -np.pi] += two_pi
                ang_diff[ang_diff > np.pi] -= two_pi

                def swap_boxes_3d_lw(boxes_3d):
                    boxes_3d_lengths = np.copy(boxes_3d[:, 3])
                    boxes_3d[:, 3] = boxes_3d[:, 4]
                    boxes_3d[:, 4] = boxes_3d_lengths
                    return boxes_3d

                pi_0_25 = 0.25 * np.pi
                pi_0_50 = 0.50 * np.pi
                pi_0_75 = 0.75 * np.pi

                # Rotate 90 degrees if difference between pi/4 and 3/4 pi
                rot_pos_90_indices = np.logical_and(pi_0_25 < ang_diff,
                                                    ang_diff < pi_0_75)
                final_pred_boxes_3d[i][rot_pos_90_indices] = \
                    swap_boxes_3d_lw(final_pred_boxes_3d[i][rot_pos_90_indices])
                final_pred_boxes_3d[i][rot_pos_90_indices, 6] += pi_0_50

                # Rotate -90 degrees if difference between -pi/4 and -3/4 pi
                rot_neg_90_indices = np.logical_and(-pi_0_25 > ang_diff,
                                                    ang_diff > -pi_0_75)
                final_pred_boxes_3d[i][rot_neg_90_indices] = \
                    swap_boxes_3d_lw(final_pred_boxes_3d[i][rot_neg_90_indices])
                final_pred_boxes_3d[i][rot_neg_90_indices, 6] -= pi_0_50

                # Flip angles if abs difference if greater than or equal to 135
                # degrees
                swap_indices = np.abs(ang_diff) >= pi_0_75
                final_pred_boxes_3d[i][swap_indices, 6] += np.pi

                # Wrap to -pi, pi
                above_pi_indices = final_pred_boxes_3d[i][:, 6] > np.pi
                final_pred_boxes_3d[i][above_pi_indices, 6] -= two_pi
        else:
            raise NotImplementedError('Parse predictions not implemented for', box_rep)

        final_corr_offsets = predictions[DtAvodModel.PRED_TOP_CORR_OFFSETS]

        final_pred_corr_boxes_3d = final_pred_boxes_3d[0].copy()
        # decoder corr offsets
        # final_corr_offsets = np.log(final_corr_offsets / (1 - final_corr_offsets))
        final_pred_corr_boxes_3d[:, 0] += final_corr_offsets[:, 0]
        final_pred_corr_boxes_3d[:, 2] += final_corr_offsets[:, 1]
        final_pred_corr_boxes_3d[:, -1] += final_corr_offsets[:, -1]

        # Append score and class index (object type)
        final_pred_softmax = predictions[DtAvodModel.PRED_TOP_CLASSIFICATION_SOFTMAX]

        corr_mark = np.zeros((final_pred_softmax[1].shape[0],
                              final_pred_corr_boxes_3d.shape[1]))
        final_pred_corr_boxes_3d = [final_pred_corr_boxes_3d, corr_mark]

        # Find max class score index
        not_bkg_scores = [pred_softmax[:, 1:] for pred_softmax in final_pred_softmax]
        final_pred_types = [np.argmax(score, axis=1) for score in not_bkg_scores]

        # Take max class score (ignoring background)
        size = len(final_pred_boxes_3d)
        predictions_and_scores = [None] * size
        for i in range(size):
            final_pred_scores = np.array([])
            frame_mark = np.ones((len(final_pred_boxes_3d[i]), 1)) * i

            for pred_idx in range(len(final_pred_boxes_3d[i])):
                all_class_scores = not_bkg_scores[i][pred_idx]
                max_class_score = all_class_scores[final_pred_types[i][pred_idx]]
                final_pred_scores = np.append(final_pred_scores, max_class_score)

            # Stack into prediction format
            predictions_and_scores[i] = np.column_stack(
                [final_pred_boxes_3d[i],
                 final_pred_scores,
                 final_pred_types[i],
                 final_pred_corr_boxes_3d[i],
                 frame_mark])

        predictions_and_scores = np.concatenate(predictions_and_scores, axis=0)

        return predictions_and_scores

    def get_avod_predicted_box_corners_and_scores(self, predictions, box_rep):

        final_corr_offsets = predictions[DtAvodModel.PRED_TOP_CORR_OFFSETS]
        final_pred_boxes_3d = predictions[DtAvodModel.PRED_TOP_PREDICTION_BOXES_3D][0]
        final_pred_corr_boxes_3d = final_pred_boxes_3d
        final_pred_corr_boxes_3d[:, :3] += final_corr_offsets[:, :3]
        final_pred_corr_boxes_3d[:, -1] += final_corr_offsets[:, -1]

        if box_rep in ['box_8c', 'box_8co']:
            final_pred_box_corners = predictions[DtAvodModel.PRED_TOP_BOXES_8C]
            #
            # final_pred_corr_box_corners = box_8c_encoder.tf_box_3d_to_box_8co(
            #                                 final_pred_corr_boxes_3d)

        elif box_rep in ['box_4c', 'box_4ca']:
            final_pred_box_corners = predictions[DtAvodModel.PRED_TOP_BOXES_4C]
            #
            # ground_plane = np.array([0, -1, 0, 1.65])
            # final_pred_corr_box_corners = box_4c_encoder.tf_box_3d_to_box_4c(
            #                                 final_pred_corr_boxes_3d, ground_plane)

        # Append score and class index (object type)
        final_pred_softmax = predictions[DtAvodModel.PRED_TOP_CLASSIFICATION_SOFTMAX]

        # get top correlation offsets
        corr_mark = np.zeros((final_pred_softmax[1].shape[0],
                              final_pred_corr_boxes_3d.shape[1]))
        final_pred_corr_box_corners = [final_pred_corr_boxes_3d, corr_mark]

        # Find max class score index
        not_bkg_scores = [pred_softmax[:, 1:]
                          for pred_softmax in final_pred_softmax]
        final_pred_types = [np.argmax(score, axis=1)
                            for score in not_bkg_scores]

        # Take max class score (ignoring background)
        size = len(final_pred_box_corners)
        predictions_and_scores = [None] * size
        for i in range(size):
            final_pred_scores = np.array([])
            frame_mark = np.ones((len(final_pred_box_corners[i]), 1)) * i

            for pred_idx in range(len(final_pred_box_corners[i])):
                all_class_scores = not_bkg_scores[i][pred_idx]
                max_class_score = all_class_scores[final_pred_types[i][pred_idx]]
                final_pred_scores = np.append(final_pred_scores, max_class_score)

            if box_rep in ['box_8c', 'box_8co']:
                final_pred_box_corners[i] = np.reshape(final_pred_box_corners[i],
                                                    [-1, 24])
                final_pred_corr_box_corners[i] = np.reshape(final_pred_corr_box_corners[i],
                                                    [-1,24])
            # Stack into prediction format
            predictions_and_scores[i] = np.column_stack(
                [final_pred_box_corners[i],
                 final_pred_scores,
                 final_pred_types[i],
                 final_pred_corr_box_corners[i],
                 frame_mark])

        predictions_and_scores = np.concatenate(predictions_and_scores, axis=0)

        return predictions_and_scores

    def run_kitti_native_eval(self, global_step):
        """Calls the kitti native C++ evaluation code.

        It first saves the predictions in kitti format. It then creates two
        child processes to run the evaluation code. The native evaluation
        hard-codes the IoU threshold inside the code, so hence its called
        twice for each IoU separately.

        Args:
            global_step: Global step of the current checkpoint to be evaluated.
        """

        # Kitti native evaluation, do this during validation
        # and when running Avod model.
        # Store predictions in kitti format
        evaluator_utils.save_predictions_in_kitti_format(
            self.model,
            self.model_config.checkpoint_name,
            self.dataset_config.data_split,
            self.eval_config.kitti_score_threshold,
            global_step,
            is_detection_single=False)

        checkpoint_name = self.model_config.checkpoint_name
        kitti_score_threshold = self.eval_config.kitti_score_threshold

        # Create a separate processes to run the native evaluation
        native_eval_proc = Process(
            target=evaluator_utils.run_kitti_native_script, args=(
                checkpoint_name, kitti_score_threshold, global_step))
        native_eval_proc_05_iou = Process(
            target=evaluator_utils.run_kitti_native_script_with_05_iou,
            args=(checkpoint_name, kitti_score_threshold, global_step))
        # Don't call join on this cuz we do not want to block
        # this will cause one zombie process - should be fixed later.
        native_eval_proc.start()
        native_eval_proc_05_iou.start()


    def run_kitti_native_tracking_eval(self, root_dir, output_dir, global_step):
        video_frames = self.video_frames
        dataset = self.model.dataset
        # flag for the exist of vaild trajectory
        os.makedirs(output_dir,exist_ok=True)
        is_empty = True
        for (video_id, frames) in video_frames.items():
            dets_for_track, dets_for_ious = encoder_tracking_dets(
                root_dir, frames, dataset,
                self.eval_config.track_lth)

            # track_iou algorithm
            tracks_finished = track_through_ious(
                dets_for_track, dets_for_ious,
                self.eval_config.track_hth,
                self.eval_config.track_liou,
                self.eval_config.track_tmin
            )
            # convert tracks into kitti format
            track_kitti_format = convert_trajectory_to_kitti_format(tracks_finished)

            if len(track_kitti_format) != 0:
                is_empty = False
            # store final result
            # create txt to store tracking predictions
            video_result_path = output_dir + video_id.zfill(4) + '.txt'
            np.savetxt(video_result_path, track_kitti_format, newline='\r\n', fmt='%s')

        # run eval script
        if not is_empty:
            eval_script_dir = output_dir + '/../../../'
            eval_script = eval_script_dir + 'evaluate_tracking.py'
            code = 'python %s %s %s' % (eval_script, eval_script_dir, global_step)
            os.system(code)
        else:
            print('There is not vaild trajectory in prediction result!')