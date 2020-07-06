import os
import copy
import collections
from distutils import dir_util

import numpy as np
import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.dt_inference_utils import convert_pred_to_kitti_format

from wavedata.tools.obj_detection.evaluation import three_d_iou


def config_setting(checkpoint_name, ckpt_indices):
    root_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
               '/predictions/final_predictions_and_scores/val/' + ckpt_indices

    # Read the config from the experiment folder
    experiment_config_path = avod.root_dir() + '/data/outputs/' + \
                             checkpoint_name + '/' + checkpoint_name + '.config'

    tracking_eval_script_dir = avod.root_dir() + '/data/outputs/' + \
                               checkpoint_name + '/predictions/kitti_tracking_native_eval/'

    tracking_output_dir = tracking_eval_script_dir + 'results/' + ckpt_indices + '/data/'

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
    dataset_config.has_labels = False
    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []
     # Build the dataset object
    dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config,
                                                     use_defaults=False)
    return dataset


def get_frames(dataset):
    video_frames = {}
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


def iou_3d(box3d_1, box3d_2):
    # convert to [ry, l, h, w, tx, ty, tz]
    box3d = box3d_1[[-2, 0, 2, 1, 3, 4, 5]]
    box3d[1:4] = 4 * box3d[1:4]
    if len(box3d_2.shape) == 1:
        boxes3d = box3d_2[[-2, 0, 2, 1, 3, 4, 5]]
    else:
        boxes3d = box3d_2[:, [-2, 0, 2, 1, 3, 4, 5]]
    iou = three_d_iou(box3d, boxes3d)
    return iou


def decode_tracking_file(root_dir, file_name, dataset, threshold=0.1):
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
    rect_pred_frame_0 = pred_frame_0.copy()
    rect_pred_frame_0[:,:7] = np_file[frame_mask_0][:, 9:-1]

    # frame_0
    pred_frame_kitti_0 = convert_pred_to_kitti_format(
        pred_frame_0, sample_name_0, dataset, threshold)
    # frame_1
    pred_frame_kitti_1 = convert_pred_to_kitti_format(
        pred_frame_1, sample_name_1, dataset, threshold)
    # rect_frame_0
    rect_pred_frame_kitti_0 = convert_pred_to_kitti_format(
        rect_pred_frame_0, sample_name_0, dataset, threshold)

    return pred_frame_kitti_0, pred_frame_kitti_1, rect_pred_frame_kitti_0


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


def generate_dets_for_dt_track(frames, root_dir):
    frames.sort()
    frame_num = len(frames)
    dets_for_track = []
    # first item is empty
    dets_for_ious = [{}]
    # create tracking list
    for i in range(frame_num):
        frame_name_0 = int(frames[i].split('_')[0][2:])
        frame_name_1 = int(frames[i].split('_')[1][2:])
        # get kitti type predicted label and offsets
        pred_frame_kitti_0, pred_frame_kitti_1, rect_pred_frame_kitti_0 = \
            decode_tracking_file(root_dir, frames[i], dataset)

        if len(pred_frame_kitti_0) == 0 and len(pred_frame_kitti_1) == 0:
            continue
        elif len(pred_frame_kitti_0) == 0:
            track_item = []
        else:
            track_item = [{'frame_id': int(frame_name_0),
                           'info': frame[:4],
                           'boxes2d': np.array(frame[4:8], dtype=np.float32),
                           'boxes3d': np.array(frame[8:-1], dtype=np.float32),
                           'rect_boxes2d': np.array(offset[4:8], dtype=np.float32),
                           'rect_boxes3d': np.array(offset[8:-1], dtype=np.float32),
                           'scores': np.array(frame[-1], dtype=np.float32)}
                          for (frame, offset) in zip(pred_frame_kitti_0,
                                                     rect_pred_frame_kitti_0)]
        iou_item = [{'frame_id': int(frame_name_1),
                     'info': frame[:4],
                     'boxes2d': np.array(frame[4:8], dtype=np.float32),
                     'boxes3d': np.array(frame[8:-1], dtype=np.float32),
                     'scores': np.array(frame[-1], dtype=np.float32)}
                    for frame in pred_frame_kitti_1]

        dets_for_track.append(track_item)
        dets_for_ious.append(iou_item)
    return dets_for_track, dets_for_ious


def track_iou(dets_for_track, dets_for_ious,
                 high_threshold, iou_threshold, t_min, ttl=3):

    tracks_active = []
    tracks_finished = []

    for frame_num, dets in enumerate(dets_for_track, start=1):
        # apply low threshold to detections
        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                ious = [iou_3d(track['trajectory'][-1]['rect_boxes3d'], x['boxes3d']) for x in dets]
                best_match_id = int(np.argmax(ious))
                if ious[best_match_id] > iou_threshold:
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
                if track['max_score'] >= high_threshold and len(track['trajectory']) >= t_min:
                    tracks_finished.append(track)

            # if track was not updated
            if track['virtual_len'] == ttl:
                track['trajectory'] = track['trajectory'][:-ttl]
                track['virtual_len'] = -1
                # finish track when the conditions are met
                if track['max_score'] >= high_threshold and len(track['trajectory']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'trajectory': [det], 'max_score': det['scores'],
                       'start_frame': frame_num-1, 'virtual_len': 0} for det in dets]
        updated_tracks = [track for track in updated_tracks if track['virtual_len'] != -1]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active if track['max_score'] >= high_threshold
                        and len(track['trajectory']) >= t_min]

    return tracks_finished

def compute_mid_frame(track):
    track_next = track[1:]
    track_pre = track[:-1]
    new_track = []
    new_track.append(track_pre[0])
    for (pre, next) in zip(track_pre, track_next):
        pre_frame_id = pre['frame_id']
        next_frame_id = next['frame_id']
        stride = next_frame_id - pre_frame_id - 1
        offsets_2d = (next['boxes2d'] - pre['boxes2d']) / (stride + 1)
        offsets_3d = next['boxes3d'] - pre['boxes3d'] / (stride + 1)
        score = max(next['scores'], pre['scores'])

        while stride > 0:
            new_item = {}
            new_item['frame_id'] = new_track[-1]['frame_id'] + 1
            new_item['info'] = new_track[-1]['info']
            new_item['boxes2d'] = new_track[-1]['boxes2d'] + offsets_2d
            new_item['boxes3d'] = new_track[-1]['boxes3d'] + offsets_3d
            new_item['scores'] = score
            new_track.append(new_item)
            stride = stride - 1
        new_track.append(next)
    return new_track


if __name__ == '__main__':
    checkpoint_name = 'pyramid_cars_with_aug_dt_5_stride_3_tracking_corr_pretrained'
    ckpt_indices = '38000'

    root_dir, tracking_output_dir, tracking_eval_script_dir, \
    dataset_config = config_setting(checkpoint_name, ckpt_indices)

    dataset = build_dataset(dataset_config)
    frame_stride = dataset.data_stride
    video_frames = get_frames(dataset)

    # copy tracking eval script to tracking_output_dir
    video_ids = video_frames.keys()
    copy_tracking_eval_script(tracking_eval_script_dir, video_ids)

    for (video_id, frames) in video_frames.items():
        dets_for_track, dets_for_ious = generate_dets_for_dt_track(frames, root_dir)

        # split dets according frame_stride
        track_stride_dets = [[] for _ in range(frame_stride)]
        ious_stride_dets  = [[] for _ in range(frame_stride)]
        for i in range(len(dets_for_track)):
            index = int(i % frame_stride)
            track_stride_dets[index].append(dets_for_track[i])
            ious_stride_dets[index].append(dets_for_ious[i])

        # track_iou algorithm
        temp_dets = track_stride_dets[0]
        temp_ious = ious_stride_dets[0]
        # add last frame
        if temp_dets[-1][0]['frame_id'] != dets_for_track[-1][0]['frame_id']:
            temp_dets.append(dets_for_track[-1])
            temp_ious.append(dets_for_ious[-1])
        tracks_finished = track_iou(temp_dets, temp_ious,
                            high_threshold=0.5, iou_threshold=0.00, t_min=3)

        # compute interframe
        if frame_stride > 1:
            new_tracks = []
            for item in tracks_finished:
                track = item['trajectory']
                new_track = compute_mid_frame(track)
                item['trajectory'] = new_track
                new_tracks.append(item)
            tracks_finished = new_tracks

        # convert tracks into kitti format
        track_kitti_format = convert_trajectory_to_kitti_format(tracks_finished)
        # store final result
        # create txt to store tracking predictions
        video_result_path = tracking_output_dir + video_id.zfill(4) + '.txt'
        np.savetxt(video_result_path, track_kitti_format, newline='\r\n', fmt='%s')
        print('store prediction results:', video_result_path)

    # run eval script
    run_kitti_tracking_script(checkpoint_name, ckpt_indices)
