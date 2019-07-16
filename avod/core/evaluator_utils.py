import sys
import os
import copy
import datetime
import subprocess
from distutils import dir_util

import numpy as np
from PIL import Image
import tensorflow as tf

import avod
from avod.core import box_3d_projector
from avod.core import summary_utils
from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection.obj_utils import ObjectLabel
from wavedata.tools.obj_detection.evaluation import three_d_iou
from avod.core.dt_inference_utils import convert_pred_to_kitti_format


def save_predictions_in_kitti_format(model,
                                     checkpoint_name,
                                     data_split,
                                     score_threshold,
                                     global_step,
                                     is_detection_single=True):
    """ Converts a set of network predictions into text files required for
    KITTI evaluation.
    """

    dataset = model.dataset
    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    # Get available prediction folders
    predictions_root_dir = avod.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'

    if is_detection_single:
        final_predictions_root_dir = predictions_root_dir + \
            '/final_predictions_and_scores/' + dataset.data_split
    else:
        final_predictions_root_dir = predictions_root_dir + \
            '/kitti_detection_predictions_and_scores/' + dataset.data_split

    final_predictions_dir = final_predictions_root_dir + \
        '/' + str(global_step)

    # 3D prediction directories
    kitti_predictions_3d_dir = predictions_root_dir + \
        '/kitti_native_eval/' + \
        str(score_threshold) + '/' + \
        str(global_step) + '/data'

    if not os.path.exists(kitti_predictions_3d_dir):
        os.makedirs(kitti_predictions_3d_dir)

    # Do conversion
    num_samples = dataset.num_samples
    num_valid_samples = 0

    print('\nGlobal step:', global_step)
    print('Converting detections from:', final_predictions_dir)

    print('3D Detections being saved to:', kitti_predictions_3d_dir)

    for sample_idx in range(num_samples):

        # Print progress
        sys.stdout.write('\rConverting {} / {}'.format(
            sample_idx + 1, num_samples))
        sys.stdout.flush()

        sample_name = dataset.sample_names[sample_idx]

        if is_detection_single:
            prediction_file = sample_name + '.txt'
        else:
            prediction_file = sample_name[0] + '.txt'

        kitti_predictions_3d_file_path = kitti_predictions_3d_dir + \
            '/' + prediction_file

        predictions_file_path = final_predictions_dir + \
            '/' + prediction_file

        # If no predictions, skip to next file
        if not os.path.exists(predictions_file_path):
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        all_predictions = np.loadtxt(predictions_file_path)

        score_filter = all_predictions[:, 7] >= score_threshold
        all_predictions = all_predictions[score_filter]

        # If no predictions, skip to next file
        if len(all_predictions) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        # Project to image space
        sample_name = prediction_file.split('.')[0]

        # Load image for truncation
        image = Image.open(dataset.get_rgb_image_path(sample_name))

        if is_detection_single:
            img_idx = int(sample_name)
            stereo_calib_p2 = calib_utils.read_calibration(dataset.calib_dir,
                                                       img_idx).p2
        else:
            img_idx = sample_name
            video_id = int(img_idx[:2])
            stereo_calib_p2 = calib_utils.read_tracking_calibration(
                dataset.calib_dir, video_id).p2

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
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        num_valid_samples += 1

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

        # Save to text files
        np.savetxt(kitti_predictions_3d_file_path, kitti_text_3d,
                   newline='\r\n', fmt='%s')

    print('\nNum valid:', num_valid_samples)
    print('Num samples:', num_samples)


def save_stack_predictions_in_kitti_format(model,
                                     checkpoint_name,
                                     data_split,
                                     score_threshold,
                                     global_step,
                                     is_detection_single=True):
    """ Converts a set of network predictions into text files required for
    KITTI evaluation.
    """

    dataset = model.dataset
    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    # Get available prediction folders
    predictions_root_dir = avod.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'

    if is_detection_single:
        final_predictions_root_dir = predictions_root_dir + \
            '/final_predictions_and_scores/' + dataset.data_split
    else:
        final_predictions_root_dir = predictions_root_dir + \
            '/kitti_detection_predictions_and_scores/' + dataset.data_split

    final_predictions_dir = final_predictions_root_dir + \
        '/' + str(global_step)

    # 3D prediction directories
    kitti_predictions_3d_dir = predictions_root_dir + \
        '/kitti_native_eval/' + \
        str(score_threshold) + '/' + \
        str(global_step) + '/data'

    if not os.path.exists(kitti_predictions_3d_dir):
        os.makedirs(kitti_predictions_3d_dir)

    # Do conversion
    num_samples = dataset.num_samples
    num_valid_samples = 0

    print('\nGlobal step:', global_step)
    print('Converting detections from:', final_predictions_dir)

    print('3D Detections being saved to:', kitti_predictions_3d_dir)

    for sample_idx in range(num_samples):

        # Print progress
        sys.stdout.write('\rConverting {} / {}'.format(
            sample_idx + 1, num_samples))
        sys.stdout.flush()

        sample_name = dataset.sample_names[sample_idx]

        if is_detection_single:
            prediction_file = sample_name + '.txt'
        else:
            prediction_file = sample_name[0] + '.txt'

        kitti_predictions_3d_file_path = kitti_predictions_3d_dir + \
            '/' + prediction_file

        predictions_file_path = final_predictions_dir + \
            '/' + prediction_file

        # If no predictions, skip to next file
        if not os.path.exists(predictions_file_path):
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        all_predictions = np.loadtxt(predictions_file_path)

        score_filter = all_predictions[:, 8] >= score_threshold
        all_predictions = all_predictions[score_filter]

        # If no predictions, skip to next file
        if len(all_predictions) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        # Project to image space
        sample_name = prediction_file.split('.')[0]

        # Load image for truncation
        image = Image.open(dataset.get_rgb_image_path(sample_name))

        if is_detection_single:
            img_idx = int(sample_name)
            stereo_calib_p2 = calib_utils.read_calibration(dataset.calib_dir,
                                                       img_idx).p2
        else:
            img_idx = sample_name
            video_id = int(img_idx[:2])
            stereo_calib_p2 = calib_utils.read_tracking_calibration(
                dataset.calib_dir, video_id).p2

        boxes = []
        image_filter = []
        for i in range(len(all_predictions)):
            box_3d = all_predictions[i, 1:8]
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
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        num_valid_samples += 1

        # To keep each value in its appropriate position, an array of zeros
        # (N, 16) is allocated but only values [4:16] are used
        kitti_predictions = np.zeros([len(boxes), 16])

        # Get object types
        all_pred_classes = all_predictions[:, 9].astype(np.int32)
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
        kitti_predictions[:, 8] = all_predictions[:, 6]
        kitti_predictions[:, 9] = all_predictions[:, 5]
        kitti_predictions[:, 10] = all_predictions[:, 4]
        # (x, y, z)
        kitti_predictions[:, 11:14] = all_predictions[:, 1:4]
        # (ry, score)
        kitti_predictions[:, 14:16] = all_predictions[:, 7:9]

        # Round detections to 3 decimal places
        kitti_predictions = np.round(kitti_predictions, 3)

        # Empty Truncation, Occlusion
        kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                     dtype=np.int32)

        # Stack 3D predictions text
        kitti_text_3d = np.column_stack([obj_types,
                                         kitti_empty_1,
                                         kitti_predictions[:, 3:16]])

        # Save to text files
        np.savetxt(kitti_predictions_3d_file_path, kitti_text_3d,
                   newline='\r\n', fmt='%s')

    print('\nNum valid:', num_valid_samples)
    print('Num samples:', num_samples)


def recovery_predictions(dataset, sample_names, predictions):
    base_name = sample_names[0]
    num = len(sample_names)
    lists = [predictions[predictions[:, -1] == i] for i in range(num)]
    lists = [list[:, :-1] for list in lists]

    for i in range(1, num):
        temp_name = sample_names[i]
        trans, matrix, delta = dataset.coordinate_transform(
                                [base_name, temp_name])
        calib = dataset.kitti_utils.get_calib(dataset.bev_source, temp_name)

        top_list = lists[i]
        if len(top_list) != 0:
            for j in range(len(top_list)):
                # convert to TrackingLabel object
                obj = ObjectLabel()
                obj.t = top_list[j][1:4]
                obj.l = top_list[j][6]
                obj.w = top_list[j][5]
                obj.h = top_list[j][4]
                obj.ry = top_list[j][7]

                obj.t = dataset.recovery_t(obj, calib, trans, matrix)
                obj.ry -= delta

                # convert back to numpy
                top_list[j][1:4] = obj.t
                top_list[j][7] = obj.ry
    return lists

def recovery_coordinate(dataset, samples_names, predictions):
    base_name = samples_names[0]
    curr_name = samples_names[1]
    trans, matrix, delta = dataset.coordinate_transform([base_name, curr_name])
    calib = dataset.kitti_utils.get_calib(dataset.bev_source, curr_name)
    if len(predictions) != 0:
        for j in range(len(predictions)):
            # convert to TrackingLabel object
            obj = ObjectLabel()
            obj.t = predictions[j][1:4]
            obj.l = predictions[j][6]
            obj.w = predictions[j][5]
            obj.h = predictions[j][4]
            obj.ry = predictions[j][7]

            obj.t = dataset.recovery_t(obj, calib, trans, matrix)
            obj.ry -= delta

            # convert back to numpy
            predictions[j][1:4] = obj.t
            predictions[j][7] = obj.ry
    return predictions

def interpolate_non_keyframe_predicitons(dataset, sample_names, predictions, threshold):
    def cal_iou(box3d_1, box3d_2):
        # convert to [ry, l, h, w, tx, ty, tz]
        box3d = box3d_1[[7, 4, 5, 6, 1, 2, 3]]
        if len(box3d_2.shape) == 1:
            boxes3d = box3d_2[[7, 4, 5, 6, 1, 2, 3]]
        else:
            boxes3d = box3d_2[:, [7, 4, 5, 6, 1, 2, 3]]
        iou = three_d_iou(box3d, boxes3d)
        return iou

    all_sample_names = dataset.create_all_sample_names(sample_names)
    num = len(all_sample_names)
    pred_lists = [predictions[predictions[:, -1] == i] for i in range(num)]
    # only one valid frame
    if num  == 1:
        # [anchor_id, x, y, z, l, w, h, r, score, type]
        final_predictions = [pred_lists[0][:, :-4]]
        return final_predictions, all_sample_names
    # two adjacent valid frame, no need for interpolation
    if num == 2:
        # convert frame 2 to it own coordinate
        final_predictions = [pred[:, :-4] for pred in pred_lists]
        final_predictions[1] = recovery_coordinate(dataset, sample_names,
                                                   final_predictions[1])
        assert len(final_predictions) == len(all_sample_names)
        return final_predictions, all_sample_names

    # more than 3 frames
    else:
        # keep box which score large than 0.1
        kept_list = [pred[np.where(pred[:, 8] > threshold)[0]] for pred in pred_lists]
        # object match
        trajectories = []
        if len(kept_list[0]) == 0:
            if len(kept_list[1]) == 0:
                final_predictions = [[] for _ in all_sample_names]
                return final_predictions, all_sample_names
            else:
                remain_obj = kept_list[1]
                for obj in remain_obj:
                    trajectories.append([[], obj])
        else:
            next_idx = [i for i in range(len(kept_list[1]))]
            for i in range(kept_list[0]):
                curr_obj = kept_list[0][i]
                temp_track = [curr_obj]
                ious = cal_iou(curr_obj, kept_list[1])
                best_match_id = int(np.argmax(ious))
                if(ious[best_match_id]) > 0:
                    temp_track.append(kept_list[1][best_match_id])
                    next_idx.remove(best_match_id)
                else:
                    temp_track.append([])
                trajectories.append(temp_track)
            if len(next_idx) != 0:
                remain_obj = kept_list[1][next_idx]
                for obj in remain_obj:
                    trajectories.append([[], obj])

        # do interpolation
        dense_trajectories = interpolate_trajectory(trajectories, num)
        # convert trajectories to predictions
        final_predictions = [[] for _ in range(num)]
        for track in dense_trajectories:
            assert len(track) == num
            for i in range(num):
                if track[i] != []:
                    final_predictions[i].append(track[i])
        # convert to numpy
        final_predictions = [np.asarray(pred) for pred in final_predictions]
        # coordinate transform
        for i in range(1, num):
            sample_names = [all_sample_names[0], all_sample_names[i]]
            final_predictions[i] = recovery_coordinate(dataset, sample_names,
                                                       final_predictions[i])

        return final_predictions, all_sample_names

def interpolate_trajectory(trajectories, num):
    dense_trajectories = []
    for track in trajectories:
        new_track = []
        if track[0] != [] and track[1] != []:
            track[0] = track[0][:-4]
            track[1] = track[1][:-4]
            new_track.append(track[0])
            # [delta_x, delta_z, delta_ry]
            offsets = track[1][[1, 3, 7]] - track[0][[1, 3, 7]]
            score = max(track[0][8], track[1][8])
            for i in range(num-2):
                new_obj = copy.deepcopy(track[0])
                new_obj[[1,3,7]] += offsets * (i + 1.0) / (num - 1)
                new_obj[8] = score
                new_track.append(new_obj)
            # append last frame obj
            track[1][8] = score
            new_track.append(track[1])
        elif track[0] == []:
            offsets = track[1][-4:-1]
            track[1] = track[1][:-4]
            d = np.sqrt(offsets[0]**2 + offsets[1]**2)
            # d > l/2, do interpolation
            if d <= track[1][4] / 2:
                ry = track[1][7]
                delta_x = d * np.cos(ry)
                delta_z = d * np.sin(ry)
                for i in range(num-1):
                    new_obj = copy.deepcopy(track[1])
                    new_obj[1] -= delta_x * (num - i - 2) / (num - 1)
                    new_obj[3] -= delta_z * (num - i - 2)  / (num - 1)
                    new_track.append(new_obj)
                new_track.append(track[1])
            else:
                # trajectory birth, pre half frame is None
                for i in range(num - 1):
                    if i <= num / 2:
                        new_track.append([])
                    else:
                        new_obj = copy.deepcopy(track[1])
                        new_track.append(new_obj)
                new_track.append(track[1])
        elif track[1] == []:
            offsets = track[0][-4:-1]
            track[0] = track[0][:-4]
            d = np.sqrt(offsets[0] ** 2 + offsets[1] ** 2)
            # d > l/2, do interpolation
            if d <= track[0][4] / 2:
                ry = track[0][7]
                delta_x = d * np.cos(ry)
                delta_z = d * np.sin(ry)
                new_track.append(track[0])
                for i in range(num - 1):
                    new_obj = copy.deepcopy(track[0])
                    new_obj[1] += delta_x * (i + 1.0) / (num - 1)
                    new_obj[3] += delta_z * (i + 1.0) / (num - 1)
                    new_track.append(new_obj)
            else:
                new_track.append(track[0])
                # trajectory dead, next half frame is None
                for i in range(num - 1):
                    if i >= num / 2:
                        new_track.append([])
                    else:
                        new_obj = copy.deepcopy(track[0])
                        new_track.append(new_obj)

        dense_trajectories.append(new_track)
    return dense_trajectories

def decode_tracking_file(root_dir, file_name, dataset, threshold):
    file_path = os.path.join(root_dir, file_name+'.txt')
    sample_name_0 = file_name.split('_')[0]
    sample_name_1 = file_name.split('_')[1]

    # file not exist
    if not os.path.exists(file_path):
        return [], [], []

    np_file = np.loadtxt(file_path, dtype=np.float32)
    frame_mask_0 = np.where(np_file[:, -1] == 0)[0]
    frame_mask_1 = np.where(np_file[:, -1] == 1)[0]
    pred_frame_0 = np_file[frame_mask_0][:, :9]
    pred_frame_1 = np_file[frame_mask_1][:, :9]

    pred_frame_kitti_0 = convert_pred_to_kitti_format(
        pred_frame_0, sample_name_0, dataset, threshold)
    pred_frame_kitti_1 = convert_pred_to_kitti_format(
        pred_frame_1, sample_name_1, dataset, threshold)

    # frame 1 kitti label after adding offsets
    corr_offsets = pred_frame_0
    corr_offsets[:, :7] = np_file[frame_mask_0][:, 9:-1]

    pred_frame_kitti_offsets = convert_pred_to_kitti_format(
        corr_offsets, sample_name_0, dataset, threshold)

    return pred_frame_kitti_0, pred_frame_kitti_1, pred_frame_kitti_offsets


def encoder_tracking_dets(root_dir, frames, dataset, threshold):
    frames.sort()
    frame_num = len(frames)
    dets_for_track = []
    dets_for_ious = [{}]
    for i in range(frame_num):
        frame_name_0 = int(frames[i].split('_')[0][2:])
        frame_name_1 = int(frames[i].split('_')[1][2:])
        # get kitti type predicted label and offsets
        pred_frame_kitti_0, pred_frame_kitti_1, frame_offsets = \
            decode_tracking_file(root_dir, frames[i], dataset, threshold)

        if len(pred_frame_kitti_0) == 0 and len(pred_frame_kitti_1) == 0:
            continue
        elif len(pred_frame_kitti_0) == 0:
            track_item = []
        else:
            track_item = [{'frame_id': str(frame_name_0),
                           'info': frame[:4],
                           'boxes2d': np.array(frame[4:8], dtype=np.float32),
                           'boxes3d': np.array(frame[8:-1], dtype=np.float32),
                           'offsets': np.array(offset[8:-1], dtype=np.float32),
                           'scores': np.array(frame[-1], dtype=np.float32)}
                          for (frame, offset) in zip(pred_frame_kitti_0, frame_offsets)]

        iou_item = [{'frame_id': str(frame_name_1),
                     'info': frame[:4],
                     'boxes2d': np.array(frame[4:8], dtype=np.float32),
                     'boxes3d': np.array(frame[8:-1], dtype=np.float32),
                     'scores': np.array(frame[-1], dtype=np.float32)}
                    for frame in pred_frame_kitti_1]

        dets_for_track.append(track_item)
        dets_for_ious.append(iou_item)

    return dets_for_track, dets_for_ious

def track_through_ious(dets_for_track, dets_for_ious, high_threshold,
                       iou_threshold, t_min):

    def iou_3d(box3d_1, box3d_2):
        # convert to [ry, l, h, w, tx, ty, tz]
        box3d = box3d_1[[-2, 0, 2, 1, 3, 4, 5]]
        if len(box3d_2.shape) == 1:
            boxes3d = box3d_2[[-2, 0, 2, 1, 3, 4, 5]]
        else:
            boxes3d = box3d_2[:, [-2, 0, 2, 1, 3, 4, 5]]
        iou = three_d_iou(box3d, boxes3d)
        return iou

    def merge_dets(dets, dets_iou):
        # merge dets_iou and dets
        merged_dets = dets
        for item1 in dets_iou:
            overlap = False
            boxes3d = item1['boxes3d']
            for item2 in dets:
                if iou_3d(boxes3d, item2['boxes3d']) > 0:
                    overlap = True
                    break
            if not overlap:
                item1['offsets'] = item1['boxes3d']
                merged_dets.append(item1)
        return merged_dets

    tracks_active = []
    tracks_finished = []

    for frame_num, dets in enumerate(dets_for_track, start=0):
        update_tracks = []
        # get frame label for iou computing
        dets_iou = dets_for_ious[frame_num]

        for track in tracks_active:
            if len(dets) > 0:
                # 1-2', 2-3, using 2' to compute iou, len(2') == len(2)??
                if len(dets_iou) != len(dets):
                    # merge dets_iou and dets
                    merged_dets = merge_dets(dets, dets_iou)
                    dets = copy.deepcopy(merged_dets)
                    dets_iou = copy.deepcopy(merged_dets)
                # get det with the highest iou
                # first frame uses offsets to compute iou
                ious = [iou_3d(track['trajectory'][-1]['offsets'], x['boxes3d'])
                        for x in dets_iou]
                best_match_id = int(np.argmax(ious))
                if ious[best_match_id] > iou_threshold:
                    track['trajectory'].append(dets[best_match_id])
                    track['max_score'] = max(track['max_score'], dets[best_match_id]['scores'])

                    update_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[best_match_id]
                    del dets_iou[best_match_id]

            # if track was not updated
            if len(update_tracks) == 0 or track is not update_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= high_threshold and len(track['trajectory']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'trajectory':[det], 'max_score':det['scores'],
                       'start_frame':frame_num} for det in dets]

        tracks_active = update_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active if track['max_score']
                        >= high_threshold and len(track['trajectory']) >= t_min]

    return tracks_finished


def convert_trajectory_to_kitti_format(trajectories):
    final_pred_label = []
    trace_len = len(trajectories)
    for id in range(trace_len):
        trace = trajectories[id]
        trajectory = trace['trajectory']
        score = trace['max_score']
        for obj in trajectory:
            frame_id = obj['frame_id']
            info     = obj['info'].tolist()
            boxes2d  = obj['boxes2d'].tolist()
            boxes3d  = obj['boxes3d'].tolist()

            label = [frame_id] + [id] + info + boxes2d + boxes3d + [score]
            final_pred_label.append(label)

    final_pred_label.sort(key = lambda obj: 100*int(obj[0])+int(obj[1]))
    final_pred_label = np.asarray(final_pred_label)
    return final_pred_label


def set_up_summary_writer(model_config,
                          sess):
    """ Helper function to set up log directories and summary
        handlers.
    Args:
        model_config: Model protobuf configuration
        sess : A tensorflow session
    """

    paths_config = model_config.paths_config

    logdir = paths_config.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logdir = logdir + '/eval'

    datetime_str = str(datetime.datetime.now())
    summary_writer = tf.summary.FileWriter(logdir + '/' + datetime_str,
                                           sess.graph)

    global_summaries = set([])
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_merged = summary_utils.summaries_to_keep(summaries,
                                                     global_summaries,
                                                     histograms=False,
                                                     input_imgs=False,
                                                     input_bevs=False)

    return summary_writer, summary_merged


def strip_checkpoint_id(checkpoint_dir):
    """Helper function to return the checkpoint index number.

    Args:
        checkpoint_dir: Path directory of the checkpoints

    Returns:
        checkpoint_id: An int representing the checkpoint index
    """

    checkpoint_name = checkpoint_dir.split('/')[-1]
    return int(checkpoint_name.split('-')[-1])


def print_inference_time_statistics(total_feed_dict_time,
                                    total_inference_time):

    # Print feed_dict time stats
    total_feed_dict_time = np.asarray(total_feed_dict_time)
    print('Feed dict time:')
    print('Min: ', np.round(np.min(total_feed_dict_time), 5))
    print('Max: ', np.round(np.max(total_feed_dict_time), 5))
    print('Mean: ', np.round(np.mean(total_feed_dict_time), 5))
    print('Median: ', np.round(np.median(total_feed_dict_time), 5))

    # Print inference time stats
    total_inference_time = np.asarray(total_inference_time)
    print('Inference time:')
    print('Min: ', np.round(np.min(total_inference_time), 5))
    print('Max: ', np.round(np.max(total_inference_time), 5))
    print('Mean: ', np.round(np.mean(total_inference_time), 5))
    print('Median: ', np.round(np.median(total_inference_time), 5))


def copy_kitti_native_code(checkpoint_name):
    """Copies and compiles kitti native code.

    It also creates neccessary directories for storing the results
    of the kitti native evaluation code.
    """

    avod_root_dir = avod.root_dir()
    kitti_native_code_copy = avod_root_dir + '/data/outputs/' + \
        checkpoint_name + '/predictions/kitti_native_eval/'

    # Only copy if the code has not been already copied over
    if not os.path.exists(kitti_native_code_copy):

        os.makedirs(kitti_native_code_copy)
        original_kitti_native_code = avod.top_dir() + \
            '/scripts/offline_eval/kitti_native_eval/'

        predictions_dir = avod_root_dir + '/data/outputs/' + \
            checkpoint_name + '/predictions/'
        # create dir for it first
        dir_util.copy_tree(original_kitti_native_code,
                           kitti_native_code_copy)
        # run the script to compile the c++ code
        script_folder = predictions_dir + \
            '/kitti_native_eval/'
        make_script = script_folder + 'run_make.sh'
        subprocess.call([make_script, script_folder])

    # Set up the results folders if they don't exist
    results_dir = avod.top_dir() + '/scripts/offline_eval/results'
    results_05_dir = avod.top_dir() + '/scripts/offline_eval/results_05_iou'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(results_05_dir):
        os.makedirs(results_05_dir)


def copy_kitti_native_tracking_code(checkpoint_name, video_ids, train_split='val'):
    from_path = avod.root_dir() + '/../scripts/offline_eval/' \
                'kitti_tracking_native_eval/python/'
    to_path = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
                '/predictions/kitti_tracking_native_eval/'

    # Only copy if the code has not been already copied over
    if not os.path.exists(to_path):
        os.makedirs(to_path)
        # copy data dir
        dir_util.copy_tree(from_path, to_path)

    # edit evaluate_tracking_seqmap
    if train_split in ['train', 'val', 'trainval']:
        source_seqmap_path = to_path + 'data/tracking/' \
                            'evaluate_tracking.seqmap.training'
    else:
        source_seqmap_path = to_path + 'data/tracking/' \
                                       'evaluate_tracking.seqmap.test'
    source_seqmap = open(source_seqmap_path, 'r').readlines()
    mask = [int(i) for i in video_ids]
    eval_map = to_path + 'data/tracking/evaluate_tracking.seqmap'
    with open(eval_map, 'w+') as map:
        for id in mask:
            map.write(source_seqmap[id])


def run_kitti_native_script(checkpoint_name, score_threshold, global_step):
    """Runs the kitti native code script."""

    eval_script_dir = avod.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'
    make_script = eval_script_dir + \
        '/kitti_native_eval/run_eval.sh'
    script_folder = eval_script_dir + \
        '/kitti_native_eval/'

    results_dir = avod.top_dir() + '/scripts/offline_eval/results/'

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_step),
                     str(checkpoint_name),
                     str(results_dir)])


def run_kitti_native_script_with_05_iou(checkpoint_name, score_threshold,
                                        global_step):
    """Runs the kitti native code script."""

    eval_script_dir = avod.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'
    make_script = eval_script_dir + \
        '/kitti_native_eval/run_eval_05_iou.sh'
    script_folder = eval_script_dir + \
        '/kitti_native_eval/'

    results_dir = avod.top_dir() + '/scripts/offline_eval/results_05_iou/'

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_step),
                     str(checkpoint_name),
                     str(results_dir)])


def run_kitti_tracking_script(checkpoint_name, global_step):
    eval_script_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
                      '/predictions/kitti_tracking_native_eval/'
    eval_script = eval_script_dir + 'evaluate_tracking.py'
    code = 'python %s %s %s' %(eval_script, eval_script_dir, global_step)
    print(code)
    os.system(code)
