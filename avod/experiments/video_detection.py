import os
import collections
import warnings
from distutils import dir_util

import numpy as np
import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from wavedata.tools.obj_detection.evaluation import three_d_iou


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
            experiment_config_path, is_training=False)

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
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                     use_defaults=False)
    return dataset

def iou_3d(box3d_1, box3d_2):
    # convert to [ry, l, h, w, tx, ty, tz]
    box3d = box3d_1[[-2, 0, 2, 1, 3, 4, 5]]
    # box3d[1:4] = 3 * box3d[1:4]
    if len(box3d_2.shape) == 1:
        boxes3d = box3d_2[[-2, 0, 2, 1, 3, 4, 5]]
    else:
        boxes3d = box3d_2[:, [-2, 0, 2, 1, 3, 4, 5]]
    iou = three_d_iou(box3d, boxes3d)
    return iou

def copy_tracking_eval_script(to_path, video_ids, train_split='eval'):
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
        video_id = sample_name[:2]
        if not video_frames.__contains__(video_id):
            video_frames[video_id] = []
        video_frames[video_id].append(sample_name)

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

def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['scores'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                ious = [iou_3d(track['trajectory'][-1]['boxes3d'],
                               x['boxes3d']) for x in dets]
                best_match_id = int(np.argmax(ious))
                if ious[best_match_id] > sigma_iou:
                    track['trajectory'].append(dets[best_match_id])
                    track['max_score'] = max(track['max_score'],
                                             dets[best_match_id]['scores'])

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
                       'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active if track['max_score'] >= sigma_h
                        and len(track['trajectory']) >= t_min]

    return tracks_finished


def restyle_track(track_kitti_format, frame_list):
    track_out = [[] for _ in frame_list]
    for obj in track_kitti_format:
        frame_id = int(obj[0])
        new_obj = {'obj_id':int(obj[1]),
                   'info':obj[2:6],
                   'boxes_2d':[float(i) for i in obj[6:10]],
                   'boxes_3d':[float(i) for i in obj[10:17]],
                   'score':float(obj[17])}
        track_out[frame_id].append(new_obj)
    return track_out


def label_interpolation(labels, stride):
    idx_infos = []
    video_len = len(labels)
    labels_out = []

    for i in range(video_len):
        if i % stride == 0:
            idx_infos.append([i])
        else:
            t = i % stride
            pre_i = i - t
            next_i = i + (stride - t)
            if not next_i < video_len:
                next_i = video_len - 1
            idx_infos.append([pre_i, next_i, t])

    for idx_info in idx_infos:
        if len(idx_info) == 1:
            frame_id = idx_info[0]
            labels_out.append(labels[frame_id])
        else:
            pre_label = labels[idx_info[0]]
            next_label = labels[idx_info[1]]
            curr_label = []
            if pre_label != []:
                if next_label == []:
                    curr_label = pre_label
                else:
                    curr_label = cal_label(pre_label, next_label, idx_info[2], stride)
            else:
                if next_label != []:
                    curr_label = next_label

            labels_out.append(curr_label)

    return labels_out


def cal_label(pre_label, next_label, inc, stride):
    new_label = []
    for pre_obj in pre_label:
        pre_obj_id = pre_obj['obj_id']
        temp_obj = {}
        for next_obj in next_label:
            next_obj_id = next_obj['obj_id']
            if pre_obj_id == next_obj_id:
                temp_obj['obj_id'] = pre_obj_id
                temp_obj['info'] = pre_obj['info']
                temp_obj['score'] = max(pre_obj['score'], next_obj['score'])
                temp_obj['boxes_2d'] = np.asarray(pre_obj['boxes_2d'],dtype=np.float32) +inc / stride * \
                                       (np.asarray(next_obj['boxes_2d'],dtype=np.float32)
                                        -np.asarray(pre_obj['boxes_2d'],dtype=np.float32))

                temp_obj['boxes_2d'] = [round(i, 3) for i in temp_obj['boxes_2d']]

                temp_obj['boxes_3d'] = np.asarray(pre_obj['boxes_3d'],dtype=np.float32) + inc / stride * \
                                       (np.asarray(next_obj['boxes_3d'],dtype=np.float32)
                                        -np.asarray(pre_obj['boxes_3d'],dtype=np.float32))

                temp_obj['boxes_3d'] = [round(i, 3) for i in temp_obj['boxes_3d']]

                new_label.append(temp_obj)
    return new_label


def store_final_result(frames, video_id, output_root):
    frame_len = len(frames)
    for i in range(frame_len):
        name = video_id + str(i).zfill(4)+ '.txt'
        if frames[i] == []:
            np.savetxt(output_root+name, [])
            continue
        output = []
        for obj in frames[i]:
            label = obj['info'].tolist() + obj['boxes_2d'] + \
                    obj['boxes_3d'] + [obj['score']]
            output.append(label)

        np.savetxt(output_root+name, output, newline='\r\n', fmt='%s')


if __name__ == '__main__':
    checkpoint_name = 'pyramid_cars_with_aug_example_trainval'
    ckpt_indices = '120000'

    root_dir, tracking_output_dir, tracking_eval_script_dir, \
    dataset_config = config_setting(checkpoint_name, ckpt_indices)

    output_root = avod.root_dir() + '/data/outputs/' + checkpoint_name +\
                  '/predictions/kitti_native_eval/'

    dataset = build_dataset(dataset_config)
    video_frames = get_frames(dataset)

    stride = 2
    output_root = output_root + '0.1_' + str(stride) + '/' + ckpt_indices + '/data/'
    os.makedirs(output_root, exist_ok=True)
    # copy tracking eval script to tracking_output_dir
    video_ids = video_frames.keys()
    copy_tracking_eval_script(tracking_eval_script_dir, video_ids)

    for (video_id, frames) in video_frames.items():
        dets_for_track = generate_dets_for_track(frames, root_dir)

        tracks_finished = track_iou(dets_for_track, sigma_l=0.1, sigma_h=0.5,
                                                    sigma_iou=0.00, t_min=3)

        # convert tracks into kitti format
        track_kitti_format = convert_trajectory_to_kitti_format(tracks_finished)

        track_new = restyle_track(track_kitti_format, frames)

        # interpolation
        track_interploated = label_interpolation(track_new,stride)

        # store result
        store_final_result(track_interploated, video_id, output_root)


