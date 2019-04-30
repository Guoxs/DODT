import os
import collections
import warnings
from distutils import dir_util

import numpy as np
import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from wavedata.tools.obj_detection.evaluation import three_d_iou
import copy

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
    box3d[1:4] = 3 * box3d[1:4]
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
            boxes2d  = obj['boxes2d'].tolist()
            boxes3d  = obj['boxes3d'].tolist()

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
            # 数据结构！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            track_item = [{'frame_id':   int(frames[i][2:]),
                           'info'    :   detection[:4],
                           'boxes2d' :   np.array(detection[4:8], dtype=np.float32),
                           'boxes3d' :   np.array(detection[8:-1], dtype=np.float32),
                           'scores'  :   np.array(detection[-1], dtype=np.float32),
                           'visual_track_times': 0}
                           for detection in pred_kitti]
        dets_for_track.append(track_item)
    return dets_for_track

def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    # input:   ()          阈值      阈值       阈值      临界时间
    tracks_active = []   # Ta
    tracks_finished = [] # Tf
    #visual_tracking = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        # frame_num enumerate构建的索引， detections_frame detections,下标从1开始
        dets = [det for det in detections_frame if det['scores'] >= sigma_l]
        #和论文里不一样，这样空间更小                    di有一项叫scores
        print("here")
        if(len(dets)>0):
            print(dets[0]['frame_id'])
        updated_tracks = []
        for track in tracks_active:          # for ti in Ta
            # 轨迹循环
            # 肯定要多else，假设没检测到时是漏检了
            if len(dets) > 0:                # 如果这一帧里有物体
                # 这里也要改，假如有轨迹没有匹配到
                # get det with highest iou
                ious = [iou_3d(track['trajectory'][-1]['boxes3d'],
                               x['boxes3d']) for x in dets]
                # iou_3d计算iou函数，输入是track的trajectory项的最后一个框的boxes3d项，
                # 和dets中每一项的boxes3d项
                best_match_id = int(np.argmax(ious))
                #算按最大匹配的那个框的下标
                if ious[best_match_id] > sigma_iou:
                #大于IOU阈值
                    visual_track_time=track['trajectory'][-1]['visual_track_times']
                    if visual_track_time==0:

                        track['trajectory'].append(dets[best_match_id]) #不用深拷贝
                        #加入轨迹
                        track['max_score'] = max(track['max_score'],
                                             dets[best_match_id]['scores'])
                        #max_score存最大分数，方便和分数阈值比较

                        #假如进入了这个if，就会update
                        updated_tracks.append(track)
                        # remove from best matching detection from detections
                        del dets[best_match_id]   # 变量解除关联，不释放内存
                    else:
                        track['trajectory'].append(dets[best_match_id])
                        #先加入的，所以索引-1是这个
                        track['max_score'] = max(track['max_score'],
                                                 dets[best_match_id]['scores'])
                        (x1,y1,z1,w1,h1,l1,r1)=dets[best_match_id]['boxes3d']
                        (x0,y0,z0,w0,h0,l0,r0) = track['trajectory'][-2]['boxes3d']
                        #diff = (x1-x0,y1-y0,z1-z0)
                        #single_gap = ((x1-x0)/(1+visual_track_time),(y1-y0)/(1+visual_track_time),(z1-z0)/(1+visual_track_time))
                        single_gap=(dets[best_match_id]['boxes3d']-track['trajectory'][-2]['boxes3d'])/(1+visual_track_time)
                        for vi in range(1,visual_track_time+1):
                            #track['trajectory'][-1-vi]['boxes3d'] =np.array([(x1+(x1-x0)/(1+visual_track_time)*(-vi),y1+(y1-y0)/(1+visual_track_time)*(-vi),z1+(z1-z0)/(1+visual_track_time)*(-vi),w1+(w1-w0)/(1+visual_track_time)*(-vi),h1+(h1-h0)/(1+visual_track_time)*(-vi),l1+(l1-l0)/(1+visual_track_time)*(-vi),r1+(r1-r0)/(1+visual_track_time)*(-vi))])
                            track['trajectory'][-1 - vi]['boxes3d'] = dets[best_match_id]['boxes3d']-vi*single_gap
                            track['trajectory'][-1-vi]['visual_track_times'] = 0

                        updated_tracks.append(track) #居然漏了。。
                        del dets[best_match_id]

                # 轨迹的终止没有考虑到
                else: #最大的没有匹配，复制上一帧
                    # 不用深拷贝：一定要

                    track['trajectory'].append(copy.deepcopy(track['trajectory'][-1]))  # 加入上一帧框
                    track['trajectory'][-1]['frame_id']=track['trajectory'][-2]['frame_id'] + 1
                    track['trajectory'][-1]['visual_track_times'] = track['trajectory'][-2]['visual_track_times'] + 1
                    if(track['trajectory'][-1]['visual_track_times'] <=3 ):
                        updated_tracks.append(track)
                    else:
                        track['trajectory']=track['trajectory'][:-3]
                        tracks_finished.append(track)
            else:
                tracks_finished.append(track)
            '''
            else:               # 这里肯定要有else因为假如没有一个框和轨迹匹配，就要用上一帧的结果
                track['trajectory'].append(copy.deepcopy(track['trajectory'][-1])) # 加入上一帧框
                track['trajectory'][-1]['frame_id'] = track['trajectory'][-2]['frame_id'] + 1
                track['trajectory'][-1]['visual_track_times'] = track['trajectory'][-2]['visual_track_times'] + 1
                if (track['trajectory'][-1]['visual_track_times'] <= 3):
                    updated_tracks.append(track)
                else:
                    track['trajectory'] = track['trajectory'][:-3]
                    tracks_finished.append(track)
            '''
            # if track was not updated
            #if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # updated_tracks == 0对应一开始的时候没有一个track，后面的对应当前的track
                # 没有被update，就是IOU没有超过阈值，没进if
                # finish track when the conditions are met
                # 没有被update并且最大的detection大于阈值，时间大于t_min，就加入已经结束的track
                # 这里肯定要改，因为可能是漏检，所以不能马上判定是结束，结束应该在最后一帧一起判定。
                # 但是又有个问题，在最后一帧一起判定会是道路一直阻塞，所以最好有个界限判定
             #   if track['max_score'] >= sigma_h and len(track['trajectory']) >= t_min:
              #      tracks_finished.append(track)

        # create new tracks
        # 这里显示了一个track的属性有trajectory，开始的框；max_score，初始化为开始框的分；
        # start_frame: 从第几帧开始 ； 这是个新轨迹的集合，对于dets中的每个剩下的det都假设是个新轨迹，
        # 由于匹配了的之前被删了就不会开启新轨迹了，所以新轨迹是别的物体开的。
        new_tracks = [{'trajectory': [det], 'max_score': det['scores'],
                       'start_frame': frame_num} for det in dets]
        # 更新的轨迹（加入了新框或没加） + 新开启的轨迹
        tracks_active = updated_tracks + new_tracks
        #visual_tracking = []
    # finish all remaining active tracks
    # 全部帧遍历完，还没结束的轨迹，强制结束，暂时不用改
    '''
    tracks_finished += [track for track in tracks_active if track['max_score'] >= sigma_h
                        and (len(track['trajectory']) >= t_min or (len(track['trajectory'])
                             < t_min and (track['trajectory'][-1]['frame_id'] == len(detections)
                                          or track['trajectory'][-1]['frame_id'] == len(detections)-1)))]
                                # 这就是出现的帧数，这个属性要改，但不是在这，因为判断漏检可以接上
                                # 以后相应的帧数要加上，所以还要有个列表记录结束的帧
                                '''
    for track in tracks_active:
        if track['max_score'] >= sigma_h:
            if len(track['trajectory']) >= t_min:
                print("1")
                #tracks_finished.append(track)
                if track['trajectory'][-1]['visual_track_times']==0:
                    tracks_finished.append(track)
                elif track['trajectory'][-1]['visual_track_times']==1:
                    track['trajectory'] = track['trajectory'][:-1]
                    tracks_finished.append(track)
                elif track['trajectory'][-1]['visual_track_times']==2:
                    track['trajectory'] = track['trajectory'][:-2]
                    tracks_finished.append(track)
                elif track['trajectory'][-1]['visual_track_times']==3:
                    track['trajectory'] = track['trajectory'][:-3]
                    tracks_finished.append(track)
            elif len(track['trajectory']) < t_min and len(detections) == track['trajectory'][-1]['frame_id'] + 1 and track['trajectory'][-1]['visual_track_times']==0:
                print("2")
                tracks_finished.append(track)
            elif len(track['trajectory']) < t_min and len(detections) == track['trajectory'][-1]['frame_id'] +1 and track['trajectory'][-1]['visual_track_times']==1:
                print("3")
                track['trajectory'] = track['trajectory'][:-1]
                tracks_finished.append(track)
            elif len(track['trajectory']) < t_min and len(detections) == track['trajectory'][-1]['frame_id'] + 1 and track['trajectory'][-1]['visual_track_times']==2:
                print("4")
                track['trajectory'] = track['trajectory'][:-2]
                tracks_finished.append(track)
    print("haha")
    print(len(detections))
    return tracks_finished


if __name__ == '__main__':
    checkpoint_name = 'pyramid_cars_with_aug_tracking2'
    ckpt_indices = '120000'

    root_dir, tracking_output_dir, tracking_eval_script_dir, \
    dataset_config = config_setting(checkpoint_name, ckpt_indices)

    dataset = build_dataset(dataset_config)
    video_frames = get_frames(dataset)

    # copy tracking eval script to tracking_output_dir
    video_ids = video_frames.keys()
    # copy_tracking_eval_script(tracking_eval_script_dir, video_ids)

    for (video_id, frames) in video_frames.items():
        dets_for_track = generate_dets_for_track(frames, root_dir)

        tracks_finished = track_iou(dets_for_track, sigma_l=0.1, sigma_h=0.5,
                                                    sigma_iou=0.00, t_min=3)
        # convert tracks into kitti format
        track_kitti_format = convert_trajectory_to_kitti_format(tracks_finished)
        # store final result
        # create txt to store tracking predictions
        video_result_path = tracking_output_dir + video_id.zfill(4) + '.txt'
        np.savetxt(video_result_path, track_kitti_format, newline='\r\n', fmt='%s')
        print('store prediction results:', video_result_path)

    # run eval script for evaluation
    run_kitti_tracking_script(checkpoint_name, ckpt_indices)
