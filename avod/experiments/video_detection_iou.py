import os
import collections
import subprocess
import sys
import warnings
from copy import deepcopy
from distutils import dir_util
from multiprocessing import Process

import numpy as np
import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.box_4c_encoder import np_box_3d_to_box_4c
from wavedata.tools.obj_detection.evaluation import three_d_iou, two_d_iou


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

def build_dataset(dataset_config):
    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)
    # dataset_config.data_split = 'val'
    # dataset_config.data_split_dir = 'training'
    dataset_config.data_split = 'val'
    dataset_config.data_split_dir = 'training'
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

def generate_dets_for_track(video_id, frames, root_dir):
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
            track_item = [{'video_id':   int(video_id),
                           'frame_id':   int(frames[i][2:]),
                           'info'    :   detection[:4],
                           'boxes2d' :   np.array(detection[4:8], dtype=np.float32),
                           'boxes3d' :   np.array(detection[8:-1], dtype=np.float32),
                           'score'   :   np.array(detection[-1], dtype=np.float32)}
                           for detection in pred_kitti]
        dets_for_track.append(track_item)
    return dets_for_track

def iou_3d(box3d_1, box3d_2):
    # convert to [ry, l, h, w, tx, ty, tz]
    box3d = box3d_1[[-1, 0, 2, 1, 3, 4, 5]]
    # box3d[1:4] = 4 * box3d[1:4]
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

    # box3d_1[3:6] = 3.8 * box3d_1[3:6]
    # box3d_2[3:6] = 3.8 * box3d_2[3:6]
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

def cal_transformed_ious(dataset, video_id, track, detection):
    box3d_1 = track['dets'][-1]['boxes3d']
    box3d_2 = detection['boxes3d']
    label_1 = box3d_to_label(box3d_1)
    label_2 = box3d_to_label(box3d_2)
    label_obj = [[label_1], [label_2]]

    pre_frame_id = track['dets'][-1]['frame_id']
    sample_name_1 = str(video_id).zfill(2) + str(pre_frame_id).zfill(4)
    sample_name_2 = str(video_id).zfill(2) + str(detection['frame_id']).zfill(4)
    sample_names = [sample_name_1, sample_name_2]

    transformed_label = dataset.label_transform(label_obj, sample_names)

    trans_box3d_2 = label_to_box3d(transformed_label[-1][0])

    trans_iou_2d = iou_3d(box3d_1, trans_box3d_2)

    return trans_iou_2d

def get_boundary_dis(det):
    box3d = det['boxes3d']
    x, z = box3d[3], box3d[5]
    distances = [abs(x-z)/np.sqrt(2), abs(40-x), abs(70-z),
                 abs(x+40), abs(x+z)/np.sqrt(2)]
    min_dis = min(distances)
    return min_dis

def inside(det):
    box3d = det['boxes3d']
    x, z = box3d[3], box3d[5]
    z_inside = (z > 0) & (z < 70)
    x_inside = (x > -40) & (x < 40)
    if x_inside and z_inside:
        if (z + 1.3*x) > 0 and (z - 1.3*x) > 0:
            return True
    return False

def get_absolute_speed(dataset, pre_det, next_det):
    box3d_1 = pre_det['boxes3d']
    box3d_2 = next_det['boxes3d']
    label_1 = box3d_to_label(box3d_1)
    label_2 = box3d_to_label(box3d_2)
    label_obj = [[label_1], [label_2]]

    video_id = pre_det['video_id']

    sample_name_1 = str(video_id).zfill(2) + str(pre_det['frame_id']).zfill(4)
    sample_name_2 = str(video_id).zfill(2) + str(next_det['frame_id']).zfill(4)
    sample_names = [sample_name_1, sample_name_2]

    transformed_label = dataset.label_transform(label_obj, sample_names)

    trans_box3d_2 = label_to_box3d(transformed_label[-1][0])

    speed = trans_box3d_2[[3, 5, 6]] - box3d_1[[3, 5, 6]]

    if abs(speed[-1]) > np.pi / 4:
        speed[-1] = 0

    return speed

def update_with_speed(dataset, speed, pre_det, next_id):
    pre_box3d = pre_det['boxes3d']
    pre_label = box3d_to_label(pre_box3d)
    next_box3d = pre_box3d
    next_box3d[[3,5,6]] += speed
    next_label = box3d_to_label(next_box3d)

    labels = [[pre_label], [next_label]]

    pre_id = pre_det['frame_id']
    video_id = pre_det['video_id']
    sample_name_1 = str(video_id).zfill(2) + str(pre_id).zfill(4)
    sample_name_2 = str(video_id).zfill(2) + str(next_id).zfill(4)
    sample_names = [sample_name_2, sample_name_1]

    transformed_label = dataset.label_transform(labels, sample_names)

    trans_next_box3d = label_to_box3d(transformed_label[-1][0])
    return trans_next_box3d

def interpolate_det(dataset, track, next_det, frame_num):
    pre_det = track['dets'][-1]
    pre_id = pre_det['frame_id']
    next_id = next_det['frame_id']
    stride = int(next_id) - int(pre_id)
    if stride == 0:
        print('Error in frames!')
        return
    if stride == 1:
        return track
    for i in range(1, stride):
        new_det = deepcopy(pre_det)
        new_det['frame_id'] = pre_id + i
        new_det['boxes2d'] += i / stride * (next_det['boxes2d'] - pre_det['boxes2d'])
        # [x,y,z]
        new_det['boxes3d'][3:6] += i / stride * (next_det['boxes3d'][3:6]
                                                   - pre_det['boxes3d'][3:6])
        if next_det['boxes3d'][-1] * pre_det['boxes3d'][-1] > 0:
            new_det['boxes3d'][-1] += i / stride * (next_det['boxes3d'][-1] -
                                                    pre_det['boxes3d'][-1])
        else:
            new_det['boxes3d'][-1] = next_det['boxes3d'][-1]
        track['dets'].append(new_det)

    # update speed
    track['speed'] = next_det['boxes3d'][[3, 5, 6]] - \
                     pre_det['boxes3d'][[3, 5, 6]]

    # if int(next_det['frame_id']) < frame_num:
    #     speed = get_absolute_speed(dataset, pre_det, next_det)
    #     track['speed'] = speed / stride
    return track

def extend_track_start(trajectories, stride):
    for track in trajectories:
        if len(track['dets']) < 2:
            continue
        first_det = track['dets'][0]
        first_id = first_det['frame_id']
        if not inside(first_det):
            continue
        if first_id == 0:
            continue
        speed = track['dets'][1]['boxes3d'][[3,5, 6]] - \
                first_det['boxes3d'][[3,5, 6]]

        for i in range(stride):
            pre_det = deepcopy(track['dets'][0])
            pre_det['frame_id'] -= 1
            if pre_det['frame_id'] < 0:
                break
            pre_det['boxes3d'][[3, 5, 6]] -= speed
            track['dets'].insert(0, pre_det)
    return trajectories

def extend_track_end(trajectories, stride, frame_num):
    for track in trajectories:
        if len(track['dets']) < 2:
            continue
        last_det = track['dets'][-1]
        last_id = last_det['frame_id']
        if not inside(last_det):
            continue
        if last_id >= frame_num-1:
            continue
        for i in range(stride):
            next_det = deepcopy(track['dets'][-1])
            next_det['frame_id'] += 1
            next_det['boxes3d'][[3, 5, 6]] += track['speed']
            track['dets'].append(next_det)
    return trajectories

def predict_det(track, tracks_finished, frame_num, stride, sigma_h, t_min, extend_len):
    speed = track['speed']
    is_update = True
    for i in range(stride):
        if track['extend_len'] < extend_len:
            next_det = deepcopy(track['dets'][-1])
            # video end
            # if not next_det['frame_id']+1 < frame_num:
            #     if track['max_score'] >= sigma_h and len(track['dets']) >= t_min:
            #         tracks_finished.append(track)
            #     is_update = False
            #     break
            #
            # next_det['boxes3d'] = update_with_speed(dataset, speed,
            #                                         next_det, next_det['frame_id']+1)
            next_det['frame_id'] += 1
            next_det['boxes3d'][[3, 5, 6]] += speed
            track['extend_len'] += 1
            track['dets'].append(next_det)
        else:
            track['dets'] = track['dets'][:-extend_len]
            if track['max_score'] >= sigma_h and len(track['dets']) >= t_min:
                tracks_finished.append(track)
            is_update = False
            break
    if is_update:
        return track, tracks_finished
    else:
        return None, tracks_finished


def update_dierection(track, det):
    directions = track['direction']
    angle = det['boxes3d'][-1]
    flag = 1 if angle > 0 else -1
    track['direction'].append(flag)
    if len(directions) >= 3:
        if sum(directions) > 0:
            det['boxes3d'][-1] = abs(angle)
        else:
            det['boxes3d'][-1] = -abs(angle)
    return track, det


def interpolate_by_track(dataset, video_id, detections, stride, frame_num,
                         sigma_l, sigma_h, sigma_iou, t_min, extend_len):
    tracks_active = []
    tracks_finished = []
    for frame_id, frame in enumerate(detections):
        dets = [det for det in frame if det['score'] >= sigma_l]
        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                ious = [cal_transformed_ious(dataset, video_id, track, x) for x in dets]
                # ious = [iou_2d(track['dets'][-1]['boxes3d'], x['boxes3d']) for x in dets]
                best_match_id = int(np.argmax(ious))
                # has det in next frame matches
                if ious[best_match_id] > sigma_iou:
                    if track['extend_len'] > 0: track['extend_len'] = 0
                    # update and correct direction
                    # track, dets[best_match_id] = update_dierection(track, dets[best_match_id])
                    # interpolate intermediate dets and update speed
                    track = interpolate_det(dataset, track, dets[best_match_id], frame_num)
                    # append next dets
                    track['dets'].append(dets[best_match_id])
                    track['max_score'] = max(track['max_score'], dets[best_match_id]['score'])

                    updated_tracks.append(track)
                    # remove from best matching detection from detections
                    del dets[best_match_id]
                # not has any det in next frame matches, means trajectory end or noise
                else:
                    track, tracks_finished = predict_det(track, tracks_finished, frame_num, stride,
                                                         sigma_h, t_min, extend_len)
                    if track is not None:
                        updated_tracks.append(track)
            else:
                # dets is empty, update track as usual
                track, tracks_finished = predict_det(track, tracks_finished, frame_num, stride,
                                                     sigma_h, t_min, extend_len)
                if track is not None:
                    updated_tracks.append(track)

        # create new tracks
        new_tracks = [{'dets': [det], 'direction': [1 if det['boxes3d'][-1] > 0 else -1],
                       'max_score': det['score'], 'speed': [0.0, 0.0, 0.0], 'extend_len': 0} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active if track['max_score'] >= sigma_h
                                and len(track['dets']) >= t_min]

    # improve direction

    # extend trajectory start
    tracks_finished = extend_track_start(tracks_finished, stride)

    # extend trajectory end
    tracks_finished = extend_track_end(tracks_finished, stride, frame_num)

    return tracks_finished

def convert_trajectory_to_kitti_format(trajectories):
    final_pred_label = []
    trace_len = len(trajectories)
    for id in range(trace_len):
        trace = trajectories[id]
        trajectory = trace['dets']
        score = trace['max_score']
        for obj in trajectory:
            frame_id = obj['frame_id']
            info     = obj['info'].tolist()
            boxes2d  = [round(i, 3) for i in obj['boxes2d']]
            boxes3d  = [round(i, 3) for i in obj['boxes3d']]

            label = [frame_id] + [id] + info + boxes2d + boxes3d + [score]
            final_pred_label.append(label)

    final_pred_label.sort(key = lambda obj: 100*int(obj[0])+int(obj[1]))
    final_pred_label = np.asarray(final_pred_label)
    return final_pred_label

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
    checkpoint_name = 'pyramid_cars_with_aug_dt_5_tracking_corr_pretrained_new'
    ckpt_indices = '7000'

    stride = 1

    kitti_score_threshold = '0.1_guoxs_' + str(stride)

    root_dir, tracking_output_dir, tracking_eval_script_dir, \
    dataset_config = config_setting(checkpoint_name, ckpt_indices)

    output_root = avod.root_dir() + '/data/outputs/' + checkpoint_name +\
                  '/predictions/kitti_native_eval/'

    dataset_config.data_stride = stride
    dataset = build_dataset(dataset_config)
    video_frames = get_frames(dataset)

    output_root = output_root + kitti_score_threshold + '/' + ckpt_indices + '/data/'
    os.makedirs(output_root, exist_ok=True)
    # copy tracking eval script to tracking_output_dir
    video_ids = video_frames.keys()
    copy_tracking_eval_script(tracking_eval_script_dir, video_ids)

    for (video_id, frames) in video_frames.items():
        frame_num = int(frames[-1][2:]) + 1
        dets_for_track = generate_dets_for_track(video_id, frames, root_dir)
        tracks_finished = interpolate_by_track(dataset, video_id, dets_for_track, stride, frame_num,
                                    sigma_l=0.1, sigma_h=0.5, sigma_iou=0.1, t_min=3, extend_len=3)

        # convert tracks into kitti format
        track_kitti_format = convert_trajectory_to_kitti_format(tracks_finished)

        # store tracking result
        # create txt to store tracking predictions
        video_result_path = tracking_output_dir + video_id.zfill(4) + '.txt'
        np.savetxt(video_result_path, track_kitti_format, newline='\r\n', fmt='%s')
        print('store prediction results:', video_result_path)

        # store detection result
        track_new = restyle_track(track_kitti_format, frames)
        print('\nStoring video id: %s' % video_id)
        store_final_result(track_new, video_id, output_root)

    # run eval script for tracking evaluation
    run_kitti_tracking_script(checkpoint_name, ckpt_indices)

    # Create a separate processes to run the native evaluation
    native_eval_proc = Process(target=run_kitti_native_script,
                               args=(checkpoint_name, kitti_score_threshold, ckpt_indices))

    native_eval_proc_05_iou = Process(target=run_kitti_native_script_with_05_iou,
                                      args=(checkpoint_name, kitti_score_threshold, ckpt_indices))
    # Don't call join on this cuz we do not want to block
    # this will cause one zombie process - should be fixed later.
    native_eval_proc.start()
    native_eval_proc_05_iou.start()
