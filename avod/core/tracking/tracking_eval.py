import os
import collections
import warnings
from distutils import dir_util

import numpy as np
import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from wavedata.tools.obj_detection.evaluation import three_d_iou, two_d_iou
from avod.core.box_4c_encoder import np_box_3d_to_box_4c


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
    dataset_config.data_split = 'val'
    dataset_config.data_split_dir = 'training'
    # dataset_config.data_split = 'test'
    # dataset_config.data_split_dir = 'testing'
    dataset_config.has_labels = False
    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []
     # Build the dataset object
    dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config,
                                                     use_defaults=False)
    return dataset

def iou_3d(box3d_1, box3d_2):
    # convert to [ry, l, h, w, tx, ty, tz]
    box3d = box3d_1[[-1, 2, 0, 1, 3, 4, 5]]
    # box3d[1:4] = 1.5 * box3d[1:4]
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
        trajectory = trace['trajectory']
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

def get_frames(dataset):
    video_frames = {}
    sample_names = dataset.sample_names
    for sample_name in sample_names:
        video_id = sample_name[0][:2]
        if not video_frames.__contains__(video_id):
            video_frames[video_id] = []
        video_frames[video_id].append(sample_name[0])

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
            track_item = [{'frame_id':   int(frames[i][2:]),
                           'info'    :   detection[:4],
                           'boxes2d' :   np.array(detection[4:8], dtype=np.float32),
                           'boxes3d' :   np.array(detection[8:-1], dtype=np.float32),
                           'scores'  :   np.array(detection[-1], dtype=np.float32)}
                           for detection in pred_kitti]
        dets_for_track.append(track_item)
    return dets_for_track

def track_iou(dataset, video_id, detections, sigma_l, sigma_h, sigma_iou, t_min):
    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=0):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['scores'] >= sigma_l]
        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                ious = [cal_transformed_ious(dataset, video_id, track['trajectory'][-1], x) for x in dets]
                # ious = [iou_3d(track['trajectory'][-1]['boxes3d'], x['boxes3d']) for x in dets]
                # print(ious)

                best_match_id = int(np.argmax(ious))
                if ious[best_match_id] > sigma_iou:
                    track['trajectory'].append(dets[best_match_id])
                    track['max_score'] = max(track['max_score'], dets[best_match_id]['scores'])
                    updated_tracks.append(track)
                    # remove from best matching detection from detections
                    del dets[best_match_id]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['trajectory']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'trajectory': [det], 'max_score': det['scores'],
                       'start_frame': frame_num-1} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active if track['max_score'] >= sigma_h
                        and len(track['trajectory']) >= t_min]

    return tracks_finished



def track_iou_v2(dataset, video_id, detections, sigma_l, sigma_h, sigma_iou, t_min, ttl=3):
    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['scores'] >= sigma_l]
        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                # ious = [iou_3d(track['trajectory'][-1]['boxes3d'], x['boxes3d']) for x in dets]
                ious = [cal_transformed_ious(dataset, video_id, track['trajectory'][-1], x) for x in dets]

                best_match_id = int(np.argmax(ious))
                if ious[best_match_id] > sigma_iou:
                    # convert virtual dets to valid dets
                    if track['virtual_len'] != 0:
                        t = track['virtual_len']
                        visual_dets = track['trajectory'][-t:]
                        # update virtual dets boxes coordinate
                        next_det = dets[best_match_id]
                        for i in range(t):
                            visual_dets[i]['boxes2d'] += (i+1)/(t+1)*(next_det['boxes2d']
                                                                      - visual_dets[i]['boxes2d'])
                            visual_dets[i]['boxes3d'] += (i+1)/(t+1)*(next_det['boxes3d']
                                                                      - visual_dets[i]['boxes3d'])

                        track['trajectory'][-t:] = visual_dets
                        # update virtual_len
                        track['virtual_len'] = 0

                    track['trajectory'].append(dets[best_match_id])
                    track['max_score'] = max(track['max_score'], dets[best_match_id]['scores'])
                    updated_tracks.append(track)
                    # remove from best matching detection from detections
                    del dets[best_match_id]
                else:
                    # no match det, add virtual det
                    if track['virtual_len'] < ttl:
                        visual_det = track['trajectory'][-1].copy()
                        visual_det['frame_id'] += 1
                        track['virtual_len'] += 1
                        track['trajectory'].append(visual_det)
                        updated_tracks.append(track)
            else:
                # no match det, add virtual det
                if track['virtual_len'] < ttl:
                    visual_det = track['trajectory'][-1].copy()
                    visual_det['frame_id'] += 1
                    track['virtual_len'] += 1
                    track['trajectory'].append(visual_det)
                    updated_tracks.append(track)

            if len(updated_tracks) == 0:
                track['virtual_len'] = -1
                if track['max_score'] >= sigma_h and len(track['trajectory']) >= t_min:
                    tracks_finished.append(track)

            # if track was not updated
            if track['virtual_len'] == ttl:
                track['trajectory'] = track['trajectory'][:-ttl]
                track['virtual_len'] = -1
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['trajectory']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'trajectory': [det], 'max_score': det['scores'],
                       'start_frame': frame_num-1, 'virtual_len': 0} for det in dets]
        updated_tracks = [track for track in updated_tracks if track['virtual_len'] != -1]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active if track['max_score'] >= sigma_h
                        and len(track['trajectory']) >= t_min]

    return tracks_finished


if __name__ == '__main__':
    checkpoint_name = 'pyramid_cars_with_aug_dt_5_tracking_corr_pretrained_new'
    ckpt_indices = '7000'

    root_dir, tracking_output_dir, tracking_eval_script_dir, \
    dataset_config = config_setting(checkpoint_name, ckpt_indices)

    dataset_config.data_stride = 1

    dataset = build_dataset(dataset_config)
    video_frames = get_frames(dataset)

    # copy tracking eval script to tracking_output_dir
    video_ids = video_frames.keys()
    copy_tracking_eval_script(tracking_eval_script_dir, video_ids)

    for (video_id, frames) in video_frames.items():
        dets_for_track = generate_dets_for_track(frames, root_dir)

        tracks_finished = track_iou_v2(dataset, video_id, dets_for_track, sigma_l=0.1,
                                    sigma_h=0.5, sigma_iou=0.1, t_min=3)

        # convert tracks into kitti format
        track_kitti_format = convert_trajectory_to_kitti_format(tracks_finished)
        # store final result
        # create txt to store tracking predictions
        video_result_path = tracking_output_dir + video_id.zfill(4) + '.txt'
        np.savetxt(video_result_path, track_kitti_format, newline='\r\n', fmt='%s')
        print('store prediction results:', video_result_path)

    # run eval script for evaluation
    run_kitti_tracking_script(checkpoint_name, ckpt_indices)
