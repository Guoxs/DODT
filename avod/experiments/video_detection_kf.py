import os
import collections
import subprocess
import sys
import time
import warnings
from distutils import dir_util
from collections import deque
from multiprocessing import Process

import copy
import numpy as np
import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from wavedata.tools.obj_detection.evaluation import three_d_iou, two_d_iou
from avod.core.box_4c_encoder import np_box_3d_to_box_4c

from avod.utils.kalman_tracker import Tracker


def config_setting(checkpoint_name, ckpt_indices):
    root_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
               '/predictions/kitti_native_eval/0.1/' + ckpt_indices + '/data/'

    # Read the config from the experiment folder
    experiment_config_path = avod.root_dir() + '/data/outputs/' + \
                             checkpoint_name + '/' + checkpoint_name + '.config'

    tracking_eval_script_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name \
                               + '/predictions/kitti_tracking_native_eval/'

    tracking_output_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
                          '/predictions/kitti_tracking_native_eval/results/' + \
                          ckpt_indices + '/data/'

    os.makedirs(tracking_output_dir, exist_ok=True)

    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            experiment_config_path, dataset_name='', is_training=False)

    return root_dir, tracking_output_dir, tracking_eval_script_dir, dataset_config

def build_dataset(dataset_config, data_split='val'):
    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)
    if data_split == 'val':
        dataset_config.data_split = 'val'
        dataset_config.data_split_dir = 'training'
    else:
        dataset_config.data_split = 'test'
        dataset_config.data_split_dir = 'testing'
    dataset_config.has_labels = False
    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []
     # Build the dataset object
    dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config,
                                                     use_defaults=False)
    return dataset

def run_kitti_native_script(checkpoint_name, score_threshold, global_step):
    """Runs the kitti native code script."""

    eval_script_dir = avod.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'
    make_script = eval_script_dir + \
        '/kitti_native_eval/run_eval.sh'
    script_folder = eval_script_dir + \
        '/kitti_native_eval/'

    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_step),
                     str(checkpoint_name)])

def run_kitti_native_script_with_05_iou(checkpoint_name, score_threshold,
                                        global_step):
    """Runs the kitti native code script."""

    eval_script_dir = avod.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'
    make_script = eval_script_dir + \
        '/kitti_native_eval/run_eval_05_iou.sh'
    script_folder = eval_script_dir + \
        '/kitti_native_eval/'

    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_step),
                     str(checkpoint_name)])

def iou_3d(box3d_1, box3d_2):
    # convert to [ry, l, h, w, tx, ty, tz]
    box3d = box3d_1[[-1, 2, 0, 1, 3, 4, 5]]
    if len(box3d_2.shape) == 1:
        boxes3d = box3d_2[[-1, 2, 0, 1, 3, 4, 5]]
    else:
        boxes3d = box3d_2[:, [-1, 2, 0, 1, 3, 4, 5]]
    iou = three_d_iou(box3d, boxes3d)
    return iou

def iou_2d(box3d_1, box3d_2):
    plane = np.asarray([0,-1,0,1.65])
    # convert to [tx, ty, tz, l, w, h, ry]
    box3d_1 = box3d_1[[3, 4, 5, 2, 1, 0, 6]]
    box3d_2 = box3d_2[[3, 4, 5, 2, 1, 0, 6]]

    # [x1, x2, x3, x4, z1, z2, z3, z4, h1, h2]
    box4c_1 = np_box_3d_to_box_4c(box3d_1, plane)
    box4c_2 = np_box_3d_to_box_4c(box3d_2, plane)

    box2d_1 = [np.min(box4c_1[:4]), np.max(box4c_1[4:8]),
               np.max(box4c_1[:4]), np.min(box4c_1[4:8])]
    box2d_1 = np.asarray(box2d_1)

    box2d_2 = [np.min(box4c_2[:4]), np.max(box4c_2[4:8]),
               np.max(box4c_2[:4]), np.min(box4c_2[4:8])]
    box2d_2 = np.asarray(box2d_2)
    box2d_2 = box2d_2[np.newaxis, :]
    iou = two_d_iou(box2d_1, box2d_2)
    return iou[0]

def box3d_to_label(box3d):
    from wavedata.tools.obj_detection.tracking_utils import TrackingLabel
    # boxes3d [l, w, h, tx, ty, tz, ry]
    label = TrackingLabel()
    label.l = box3d[2]
    label.w = box3d[1]
    label.h = box3d[0]
    label.t = (box3d[3], box3d[4], box3d[5])
    label.ry = box3d[6]
    return label

def label_to_box3d(label):
    box3d = [label.h, label.w, label.l, label.t[0],
            label.t[1], label.t[2], label.ry]
    box3d = np.asarray(box3d)
    return box3d


def cal_transformed_ious(dataset, video_id, item1, item2):
    box3d_1 = item1['boxes3d']
    box3d_2 = item2['boxes3d']
    label_1 = box3d_to_label(box3d_1)
    label_2 = box3d_to_label(box3d_2)
    label_obj = [[label_1], [label_2]]

    sample_name_1 = str(video_id).zfill(2) + str(item1['frame_id']).zfill(4)
    sample_name_2 = str(video_id).zfill(2) + str(item2['frame_id']).zfill(4)
    sample_names = [sample_name_1, sample_name_2]

    transformed_label = dataset.label_transform(label_obj, sample_names)

    trans_box3d_2 = label_to_box3d(transformed_label[-1][0])

    trans_iou_3d = iou_3d(box3d_1, trans_box3d_2)

    return trans_iou_3d

def copy_tracking_eval_script(to_path, video_ids, train_split='val'):
    from_path = avod.root_dir() + '/../scripts/offline_eval/' \
                'kitti_tracking_native_eval/python/'
    os.makedirs(to_path, exist_ok=True)
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

def run_kitti_tracking_script(checkpoint_name, global_step):
    eval_script_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
                      '/predictions/kitti_tracking_native_eval/'
    eval_script = eval_script_dir + 'evaluate_tracking.py'
    code = 'python %s %s %s' %(eval_script, eval_script_dir, global_step)
    print(code)
    os.system(code)

def convert_trajectory_to_kitti_format(trajectories):
    final_pred_label = []
    trace_len = len(trajectories)
    for id in range(trace_len):
        trace = trajectories[id]
        dets = trace.dets
        scores = [det['scores'] for det in dets]
        score = max(scores)
        for obj in dets:
            frame_id = obj['frame_id']
            info     = obj['info'].tolist()
            boxes2d  = [round(i, 3) for i in obj['boxes2d']]
            boxes3d  = [round(i, 3) for i in obj['boxes3d']]

            label = [frame_id] + [id] + info + boxes2d + boxes3d + [score]
            final_pred_label.append(label)

    final_pred_label.sort(key = lambda obj: int(obj[0]))
    final_pred_label = np.asarray(final_pred_label)
    return final_pred_label

def get_frames(dataset):
    video_frames = {}
    sample_names = dataset.sample_names
    for sample_name in sample_names:
        video_id = sample_name[0][:2]
        if not video_frames.__contains__(video_id):
            video_frames[video_id] = set()
        video_frames[video_id].add(sample_name[0])
        video_frames[video_id].add(sample_name[1])

    for (id, frames) in video_frames.items():
        video_frames[id] = list(frames)
        video_frames[id].sort()

    video_frames = collections.OrderedDict(sorted(video_frames.items(),
                                                  key=lambda obj: obj[0]))
    return video_frames

def generate_dets_for_track(frames, root_dir):
    frames.sort()
    frame_num = len(frames)
    dets_for_track = []
    for i in range(frame_num):
        file_path = os.path.join(root_dir, frames[i]+'.txt')
        if not os.path.exists(file_path):
            return []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_kitti = np.loadtxt(file_path, dtype=np.str)

        if len(pred_kitti) == 0:
            track_item = []
        else:
            if len(pred_kitti.shape) == 1:
                pred_kitti = np.expand_dims(pred_kitti, axis=0)
            track_item = [{'frame_id'  :   int(frames[i][2:]),
                           'info'      :   detection[:4],
                           'is_virtual':   False,
                           'boxes2d'   :   np.array(detection[4:8], dtype=np.float32),
                           'boxes3d'   :   np.array(detection[8:-1], dtype=np.float32),
                           'scores'    :   np.array(detection[-1], dtype=np.float32)}
                           for detection in pred_kitti]
        dets_for_track.append(track_item)
    return dets_for_track


def inside(det):
    box3d = det['boxes3d']
    x, z = box3d[3], box3d[5]
    z_inside = (z > 0) & (z < 70)
    x_inside = (x > -40) & (x < 40)
    if x_inside and z_inside:
        if (z + 1.3*x) > 0 and (z - 1.3*x) > 0:
            return True
    return False


def correct_direction(track, new_det):
    dets = track.dets
    directions = []
    for det in dets:
        angle = 1 if (det['boxes3d'][-1] > 0) else -1
        directions.append(angle)

    if len(directions) >= 3:
        if sum(directions) > 0:
            new_det['boxes3d'][-1] = abs(new_det['boxes3d'][-1])
            for det in dets:
                det['boxes3d'][-1] = abs(det['boxes3d'][-1])
        else:
            new_det['boxes3d'][-1] = - abs(new_det['boxes3d'][-1])
            for det in dets:
                det['boxes3d'][-1] = - abs(det['boxes3d'][-1])
    track.dets = dets
    return track, new_det


def assign_detections_to_trackers(dataset, video_id, trackers,
                                  detections, iou_threshold=0.1):
    from sklearn.utils.linear_assignment_ import linear_assignment

    if len(trackers) == 0:
        if len(detections) == 0:
            return np.asarray([]), [], []
        else:
            unmatched_detections = [i for i in range(len(detections))]
            return np.asarray([]), unmatched_detections, []
    else:
        if len(detections) == 0:
            unmatched_trackers = [i for i in range(len(trackers))]
            return np.asarray([]), [], unmatched_trackers

    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)

    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            IOU_mat[t, d] = cal_transformed_ious(dataset, video_id, trk, det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if (IOU_mat[m[0], m[1]] < iou_threshold):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, unmatched_detections, unmatched_trackers


def interpolation_detections(track, next_det, stride):
    pre_det = track.dets[-1]
    pre_frame_id = pre_det['frame_id']
    next_frame_id = next_det['frame_id']
    offsets_3d = (next_det['boxes3d'][[3,4,5]] -
               pre_det['boxes3d'][[3,4,5]]) / stride
    offsets_2d = (next_det['boxes2d'] - pre_det['boxes2d']) / stride
    for i in range(pre_frame_id+1, next_frame_id):
        new_det = copy.deepcopy(pre_det)
        new_det['boxes3d'][[3,4,5]] += offsets_3d
        new_det['boxes2d'] += offsets_2d
        new_det['frame_id'] = i
        new_det['scores'] = max(pre_det['scores'], next_det['scores'])
        new_det['is_virtual'] = True
        track.dets.append(new_det)
    # add next_det
    track.dets.append(next_det)
    return track


def kf_pipeline(dataset, video_id, detections, stride, frame_total, sigma_l, iou_threshold):
    '''
       Pipeline function for detection and tracking
    '''
    frame_count = 0
    tracker_list = []
    max_age = 3
    min_hits = 3
    track_id_list = deque([i for i in range(500)])

    final_tracker_list = []

    for frame_num, detections_frame in enumerate(detections):
        frame_count += 1
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['scores'] >= sigma_l]

        trackers = []

        if len(tracker_list) > 0:
            for trk in tracker_list:
                trackers.append(trk.dets[-1])

        matched, unmatched_dets, unmatched_trks = \
            assign_detections_to_trackers(dataset, video_id,
                                          trackers, dets, iou_threshold)

        # Deal with matched detections
        if matched.size > 0:
            for trk_idx, det_idx in matched:
                det = dets[det_idx]
                tmp_trk = tracker_list[trk_idx]
                # correct direction
                # tmp_trk, det = correct_direction(tmp_trk, det)
                z = det['boxes3d'][[3,4,5,5]]
                z = np.expand_dims(z, axis=0).T
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                trackers[trk_idx] = det
                # interpolation
                tmp_trk = interpolation_detections(tmp_trk, det, stride)
                # tmp_trk.dets.append(det)
                tmp_trk.box = xx
                tmp_trk.hits += 1
                tmp_trk.no_losses = 0

        # Deal with unmatched detections
        if len(unmatched_dets) > 0:
            for idx in unmatched_dets:
                # [l, w, h, tx, ty, tz, ry]
                det = dets[idx]
                z = np.expand_dims(det['boxes3d'], axis=0).T
                # [dx,dy,dz,dry]
                x = np.array([[z[3], 0, z[4], 0, z[5], 0, z[5], 0]]).T
                tmp_trk = Tracker()  # Create a new tracker
                tmp_trk.dets.append(det)
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = track_id_list.popleft()  # assign an ID for the tracker
                tracker_list.append(tmp_trk)
                trackers.append(det)

        # Deal with unmatched tracks
        if len(unmatched_trks) > 0:
            for trk_idx in unmatched_trks:
                tmp_trk = tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                # det outside the range
                if not inside(tmp_trk.dets[-1]):
                    # end track
                    tmp_trk.no_losses = max_age + 1
                else:
                    # complete next_det
                    next_det = copy.deepcopy(tmp_trk.dets[-1])
                    next_det['boxes3d'][[3,4,5,5]] = xx
                    next_det['frame_id'] += stride
                    # track end
                    if next_det['frame_id'] >= frame_total:
                        next_det['frame_id'] = frame_total - 1

                    next_det['is_virtual'] = True
                    tracker_list[trk_idx] = interpolation_detections(tmp_trk, next_det, stride)
                    # tmp_trk.dets.append(new_det)
                    trackers[trk_idx] = next_det

        # Book keeping
        deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

        for trk in deleted_tracks:
            # track_id_list.append(trk.id)
            if trk.hits >= min_hits:
                final_tracker_list.append(trk)

        tracker_list = [x for x in tracker_list if x.no_losses <= max_age]

    for trk in tracker_list:
        if trk.hits >= min_hits:
            final_tracker_list.append(trk)

    return final_tracker_list

def restyle_track(track_kitti_format, frame_list):
    frame_len = int(frame_list[-1][2:]) + 1
    track_out = [[] for _ in range(frame_len)]
    for obj in track_kitti_format:
        frame_id = int(obj[0])
        if frame_id > (frame_len - 1):
            break
        new_obj = {'obj_id':int(obj[1]),
                   'info':obj[2:6],
                   'boxes_2d':np.asarray([float(i) for i in obj[6:10]]),
                   'boxes_3d':np.asarray([float(i) for i in obj[10:17]]),
                   'score':float(obj[17])}
        track_out[frame_id].append(new_obj)
    return track_out


def store_final_result(frames, video_id, output_root):
    frame_len = len(frames)
    for i in range(frame_len):
        # Print progress
        sys.stdout.write('\rStoring {} / {}'.format(i + 1, frame_len))
        sys.stdout.flush()

        name = video_id + str(i).zfill(4)+ '.txt'
        if frames[i] == []:
            np.savetxt(output_root+name, [])
            continue
        output = []
        for obj in frames[i]:
            label = obj['info'].tolist() + obj['boxes_2d'].tolist() + \
                    obj['boxes_3d'].tolist() + [obj['score']]
            output.append(label)

        np.savetxt(output_root+name, output, newline='\r\n', fmt='%s')


if __name__ == '__main__':
    checkpoint_name = 'avod_stack_tracking_pretrained'
    ckpt_indices = '120000'
    data_split = 'val'

    stride = 1

    kitti_score_threshold = '0.1_kf_' + str(stride)

    root_dir, tracking_output_dir, tracking_eval_script_dir, \
    dataset_config = config_setting(checkpoint_name, ckpt_indices)

    output_root = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
                 '/predictions/kitti_native_eval/'

    dataset_config.data_stride = stride
    dataset = build_dataset(dataset_config, data_split)
    video_frames = get_frames(dataset)

    output_root = output_root + kitti_score_threshold + '/' + ckpt_indices + '/data/'
    os.makedirs(output_root, exist_ok=True)

    # copy tracking eval script to tracking_output_dir
    video_ids = video_frames.keys()
    copy_tracking_eval_script(tracking_eval_script_dir, video_ids)

    for (video_id, frames) in video_frames.items():
        frame_num = int(frames[-1][2:]) + 1
        dets_for_track = generate_dets_for_track(frames, root_dir)

        valid_frames = []
        for frame in frames:
            if frame != []:
                valid_frames.append(frame)

        start = time.time()

        tracks_finished = kf_pipeline(dataset, video_id, dets_for_track, stride,
                                      frame_num, sigma_l=0.1, iou_threshold=0.1)

        end = time.time()
        fps = len(valid_frames) / (end - start)
        print("FPS: \n", fps)

        # convert tracks into kitti format
        track_kitti_format = convert_trajectory_to_kitti_format(tracks_finished)
        # store final result
        # create txt to store tracking predictions
        video_result_path = tracking_output_dir + video_id.zfill(4) + '.txt'
        np.savetxt(video_result_path, track_kitti_format, newline='\r\n', fmt='%s')
        print('store prediction results:', video_result_path)

        # store detection result
        # track_new = restyle_track(track_kitti_format, frames)
        # print('\nStoring video id: %s' % video_id)
        # store_final_result(track_new, video_id, output_root)

    # run eval script for evaluation
    run_kitti_tracking_script(checkpoint_name, ckpt_indices)
    # Create a separate processes to run the native evaluation
    # native_eval_proc = Process(target=run_kitti_native_script,
    #                            args=(checkpoint_name, kitti_score_threshold, ckpt_indices))
    #
    # native_eval_proc_05_iou = Process(target=run_kitti_native_script_with_05_iou,
    #                                   args=(checkpoint_name, kitti_score_threshold, ckpt_indices))
    # # Don't call join on this cuz we do not want to block
    # # this will cause one zombie process - should be fixed later.
    # native_eval_proc.start()
    # native_eval_proc_05_iou.start()
