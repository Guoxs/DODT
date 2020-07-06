import numpy as np
from PIL import Image

from avod.core import box_3d_encoder, box_3d_projector
from avod.core.models.dt_avod_model import DtAvodModel
from wavedata.tools.core import calib_utils


def get_avod_pred(predictions, box_rep):
    """Returns the predictions and scores stacked for saving to file.

    Args:
        predictions: A dictionary containing the model outputs.
        box_rep: A string indicating the format of the 3D bounding
            boxes i.e. 'box_3d', 'box_8c' etc.

    Returns:
        predictions_and_scores: A numpy array of shape
            (number_of_predicted_boxes, 13), containing the final prediction
            boxes, orientations, scores, and types, frame no.
            [x, y, z, w, h, l, r, score, type, delta_x, delta_y, delta_z, frame_mark]
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
        raise NotImplementedError('Parse predictions not implemented for',
                                  box_rep)

    # Append score and class index (object type)
    final_pred_softmax = predictions[DtAvodModel.PRED_TOP_CLASSIFICATION_SOFTMAX]

    pred_corr_offsets = predictions[DtAvodModel.PRED_TOP_ANCHORS_CORR_OFFSETS]
    corr_mark = np.zeros((final_pred_softmax[1].shape[0],
                          pred_corr_offsets.shape[1]))
    final_pred_corr_offsets = [pred_corr_offsets, corr_mark]

    # Find max class score index
    not_bkg_scores = [pred_softmax[:, 1:]
                      for pred_softmax in final_pred_softmax]
    final_pred_types = [np.argmax(score, axis=1)
                        for score in not_bkg_scores]

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
             final_pred_corr_offsets[i],
             frame_mark])

    predictions_and_scores = np.concatenate(predictions_and_scores, axis=0)

    return predictions_and_scores


def convert_pred_to_kitti_format(all_predictions, sample_name, dataset, score_threshold):
    '''
    :param all_predictions: prediction from get_avod_pred
    :param dataset: kittiDataset
    :param score_threshold:
    :return: kitti format label
    '''
    score_filter = all_predictions[:, 7] >= score_threshold
    all_predictions = all_predictions[score_filter]
    if len(all_predictions) == 0:
        return []

    # Project to image space
    video_idx = int(sample_name[:2])
    # Load image for truncation
    image = Image.open(dataset.get_rgb_image_path(sample_name))
    stereo_calib_p2 = calib_utils.read_tracking_calibration(dataset.calib_dir,
                                                            video_idx).p2
    boxes = []
    image_filter = []
    for i in range(len(all_predictions)):
        box_3d = all_predictions[i, 0:7]
        img_box = box_3d_projector.project_to_image_space(
            box_3d, stereo_calib_p2,
            truncate=True, image_size=image.size)

        # Skip invalid boxes (outside image space)
        if img_box is None:
            image_filter.append(False)
            continue

        image_filter.append(True)
        boxes.append(img_box)

    boxes = np.asarray(boxes)
    all_predictions = all_predictions[image_filter]

    # If no predictions, skip to next file
    if len(boxes) == 0:
        return []

    # To keep each value in its appropriate position, an array of zeros
    # (N, 16) is allocated but only values [4:16] are used
    kitti_predictions = np.zeros([len(boxes), 16])

    # Get object types
    all_pred_classes = all_predictions[:, 8].astype(np.int32)
    obj_types = [dataset.classes[class_idx]
                 for class_idx in all_pred_classes]

    # Truncation and Occlusion are always empty (see below)

    # Alpha (Not computed)
    kitti_predictions[:, 3] = -10 * np.ones((len(kitti_predictions)),
                                            dtype=np.int32)

    # 2D predictions
    kitti_predictions[:, 4:8] = boxes[:, 0:4]

    # 3D predictions
    # (l, w, h)
    kitti_predictions[:, 8] = all_predictions[:, 5]
    kitti_predictions[:, 9] = all_predictions[:, 4]
    kitti_predictions[:, 10] = all_predictions[:, 3]
    # (x, y, z)
    kitti_predictions[:, 11:14] = all_predictions[:, 0:3]
    # (ry, score)
    kitti_predictions[:, 14:16] = all_predictions[:, 6:8]

    # Round detections to 3 decimal places
    kitti_predictions = np.round(kitti_predictions, 3)

    # Empty Truncation, Occlusion
    kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                 dtype=np.int32)

    # Stack 3D predictions text
    kitti_text_3d = np.column_stack([obj_types,
                                     kitti_empty_1,
                                     kitti_predictions[:, 3:16]])
    return kitti_text_3d