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

def iou_3d(box3d_1, box3d_2):
    # convert to [ry, l, h, w, tx, ty, tz]
    box3d = box3d_1[[-2, 0, 2, 1, 3, 4, 5]]
    if len(box3d_2.shape) == 1:
        boxes3d = box3d_2[[-2, 0, 2, 1, 3, 4, 5]]
    else:
        boxes3d = box3d_2[:, [-2, 0, 2, 1, 3, 4, 5]]
    iou = three_d_iou(box3d, boxes3d)
    return iou

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
    corr_offsets = np_file[frame_mask_0][:, 9:12]
    pred_frame_0[:, :3] += corr_offsets
    pred_frame_kitti_offsets = convert_pred_to_kitti_format(
        pred_frame_0, sample_name_0, dataset, threshold)

    return pred_frame_kitti_0, pred_frame_kitti_1, pred_frame_kitti_offsets


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
        

def track_iou(dets_for_track, dets_for_ious, sigma_h, sigma_iou, t_min):
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
                if ious[best_match_id] > sigma_iou:
                    track['trajectory'].append(dets[best_match_id])
                    track['max_score'] = max(track['max_score'], dets[best_match_id]['scores'])

                    update_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[best_match_id]
                    del dets_iou[best_match_id]

            # if track was not updated
            if len(update_tracks) == 0 or track is not update_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['trajectory']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'trajectory':[det], 'max_score':det['scores'],
                       'start_frame':frame_num} for det in dets]

        tracks_active = update_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active if track['max_score'] >= sigma_h
                        and len(track['trajectory']) >= t_min]

    return tracks_finished


checkpoint_name = 'pyramid_cars_with_aug_dt_tracking'
ckpt_indices = '104000'
root_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name + \
           '/predictions/final_predictions_and_scores/val/' + ckpt_indices

# Read the config from the experiment folder
experiment_config_path = avod.root_dir() + '/data/outputs/' +\
        checkpoint_name + '/' + checkpoint_name + '.config'

tracking_eval_script_dir = avod.root_dir() + '/data/outputs/' + \
    checkpoint_name + '/predictions/kitti_tracking_native_eval/'

tracking_output_dir = avod.root_dir() + '/data/outputs/' + \
    checkpoint_name + '/predictions/kitti_tracking_native_eval/results/' + \
    ckpt_indices + '/data/'

os.makedirs(tracking_output_dir, exist_ok=True)

model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            experiment_config_path, is_training=False)

dataset = build_dataset(dataset_config)

video_frames = {}
low_threshold = 0.5
sample_names = dataset.sample_names
for sample_name in sample_names:
    video_id = sample_name[0][:2]
    if not video_frames.__contains__(video_id):
        video_frames[video_id] = []
    txt_name = sample_name[0] + '_' + sample_name[1]
    video_frames[video_id].append(txt_name)

video_frames = collections.OrderedDict(sorted(video_frames.items(),
                                              key = lambda obj: obj[0]))

# copy tracking eval script to tracking_output_dir
video_ids = video_frames.keys()
copy_tracking_eval_script(tracking_eval_script_dir, video_ids)

for (key, values) in video_frames.items():
    # resort file
    values.sort()
    frame_num = len(values)
    dets_for_track = []
    # first item is empty
    dets_for_ious = [{}]
    # create tracking list
    for i in range(frame_num):
        frame_name_0 = int(values[i].split('_')[0][2:])
        frame_name_1 = int(values[i].split('_')[1][2:])
        # get kitti type predicted label and offsets
        pred_frame_kitti_0, pred_frame_kitti_1, frame_offsets = \
            decode_tracking_file(root_dir, values[i], dataset, low_threshold)

        if len(pred_frame_kitti_0) == 0 and len(pred_frame_kitti_1) == 0:
            continue
        elif len(pred_frame_kitti_0) == 0:
            track_item = []
        else:
            track_item = [{'frame_id':   str(frame_name_0),
                           'info'    :   frame[:4],
                           'boxes2d' :   np.array(frame[4:8], dtype=np.float32),
                           'boxes3d' :   np.array(frame[8:-1], dtype=np.float32),
                           'offsets' :   np.array(offset[8:-1], dtype=np.float32),
                           'scores'  :   np.array(frame[-1], dtype=np.float32)}
                           for (frame, offset) in zip(pred_frame_kitti_0, frame_offsets)]

        iou_item   = [{'frame_id':   str(frame_name_1),
                       'info'    :   frame[:4],
                       'boxes2d' :   np.array(frame[4:8], dtype=np.float32),
                       'boxes3d' :   np.array(frame[8:-1], dtype=np.float32),
                       'scores'  :   np.array(frame[-1], dtype=np.float32)}
                       for frame in pred_frame_kitti_1]

        dets_for_track.append(track_item)
        dets_for_ious.append(iou_item)

    # track_iou algorithm
    tracks_finished = track_iou(dets_for_track, dets_for_ious,
                                sigma_h=0.9, sigma_iou=0.005, t_min=3)

    # convert tracks into kitti format
    track_kitti_format = convert_trajectory_to_kitti_format(tracks_finished)
    # store final result
    # create txt to store tracking predictions
    video_result_path = tracking_output_dir + key.zfill(4) + '.txt'
    np.savetxt(video_result_path, track_kitti_format, newline='\r\n', fmt='%s')
    print('store prediction results:', video_result_path)

# run eval script
run_kitti_tracking_script(checkpoint_name, ckpt_indices)