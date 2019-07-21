import collections

import cv2
import os
import numpy as np
import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_encoder, box_4c_encoder
from wavedata.tools.obj_detection import tracking_utils

def build_dataset(eval_config, dataset_config):
    # Parse eval config
    eval_mode = eval_config.eval_mode

    # Parse dataset config
    data_split = dataset_config.data_split
    if data_split == 'train':
        dataset_config.data_split_dir = 'training'
        dataset_config.has_labels = True

    elif data_split.startswith('val'):
        dataset_config.data_split_dir = 'training'

        # Don't load labels for val split when running in test mode
        if eval_mode == 'val':
            dataset_config.has_labels = True
        elif eval_mode == 'test':
            dataset_config.has_labels = False

    elif data_split == 'test':
        dataset_config.data_split_dir = 'testing'
        dataset_config.has_labels = False

    else:
        raise ValueError('Invalid data split', data_split)

    # Convert to object to overwrite repeated fields
    dataset_config = config_builder.proto_to_obj(dataset_config)

    # Remove augmentation during evaluation
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_tracking_dataset(
        dataset_config, use_defaults=False)
    return dataset

def create_bev_map(dataset, name, image_shape, ground_plane):
    # get point
    point_cloud = dataset.kitti_utils.get_raw_point_cloud(
        dataset.bev_source, name)

    point_cloud = dataset.kitti_utils.transfer_lidar_to_camera_view(
        dataset.bev_source, name, point_cloud, image_shape)

    # convert point cloud to bev map and store
    bev_images = dataset.kitti_utils.create_bev_maps(point_cloud, ground_plane)
    height_maps = bev_images.get('height_maps')
    density_map = bev_images.get('density_map')
    bev_input = np.dstack((*height_maps, density_map))
    bev_input = np.sum(bev_input, axis=2)
    # bev_input = (bev_input - np.min(bev_input)) / np.max(bev_input)
    bev_input = np.minimum(bev_input * 255.0, 255.0)
    return bev_input


def project_label_to_bev_box(boxes3d, ground_plane):
    area_extents = [[-40, 40], [-5, 5], [0, 70]]
    bev_w, bev_h = 800, 700
    if len(boxes3d) == 0:
        return []
    box_4c = [box_4c_encoder.np_box_3d_to_box_4c(box3d, ground_plane)
              for box3d in boxes3d]
    # [x1, x2, x3, x4, z1, z2, z3, z4, h1, h2]
    box_4c = np.asarray(box_4c, dtype=np.float32)
    # [x1, x2, x3, x4, z1, z2, z3, z4]
    box_2d = box_4c[:, :8]
    # convert to image idx
    x_min = area_extents[0][0]
    z_min = area_extents[2][0]
    x_range = area_extents[0][1] - area_extents[0][0]
    z_range = area_extents[2][1] - area_extents[2][0]

    box_2d[:, :4] = bev_w * (box_2d[:, :4] - x_min) / x_range
    box_2d[:, 4:] = bev_h - bev_h * (box_2d[:, 4:] - z_min) / z_range

    bev_boxes = box_2d.astype(np.int32)
    return bev_boxes

def load_pred_box3d(label_dir, name, score_threshold):

    assert len(name) == 6, print('Sample name incorrect!')
    if os.stat(label_dir + "/%06d.txt" % int(name)).st_size == 0:
        return []

    p = np.loadtxt(label_dir + "/%06d.txt" % int(name), delimiter=' ',
                       dtype=str,
                       usecols=np.arange(start=0, step=1, stop=16))
    if len(p.shape) == 1:
        p = p[np.newaxis, :]

    # score filter
    kept_idx = p[:, 15].astype(np.float32) >= score_threshold
    p = p[kept_idx]
    boxes3d = p[:, [11,12,13,10,9,8,14]]
    boxes3d = boxes3d.astype(np.float32)
    return boxes3d

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=5):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def draw_rotate_rectangle(img, boxes_2d, line_type='normal', color=(255,255,255)):
    if len(boxes_2d.shape) == 1:
        boxes_2d = boxes_2d[np.newaxis,:]
    for box2d in boxes_2d:
        for (i, j) in zip([0,1,2,3], [1,2,3,0]):
            if line_type == 'dotted':
                drawline(img, (box2d[i], box2d[i+4]),
                     (box2d[j], box2d[j+4]), color, 1, style='dotted')
            else:
                cv2.line(img, (box2d[i], box2d[i+4]),
                     (box2d[j], box2d[j+4]), color, 1, lineType=cv2.LINE_AA)
    return img

def draw_prediction(dataset, input_dir, iter, output_dir):
    image_shape = [360, 1200]
    ground_plane = np.array([0, -1, 0, 1.65])
    sample_names = dataset.sample_names
    # dataset.num_samples
    for i in range(dataset.num_samples):
        name = sample_names[i][0]
        # get bev map
        bev_input = create_bev_map(dataset, name, image_shape, ground_plane)
        cv2.imwrite(output_dir + name + '.png', bev_input)

        # draw bev_boxes in img
        img = cv2.imread(output_dir + name + '.png')
        cv2.putText(img,'frame No. ' + str(name[2:]), (50, 120),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        # get label
        gt_label = tracking_utils.read_labels(dataset.label_dir, name)
        gt_label = dataset.kitti_utils.filter_labels(gt_label)
        gt_boxes3d = [box_3d_encoder.object_label_to_box_3d(obj) for obj in gt_label]
        # project label to bev
        gt_bev_boxes = project_label_to_bev_box(gt_boxes3d, ground_plane)

        # draw gt bev_box
        if len(gt_bev_boxes) > 0:
            cv2.line(img, (50,50), (100,50), (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(img, 'gt', (110, 50), cv2.FONT_HERSHEY_COMPLEX,
                                        0.5, (0, 255, 0), 1)
            img = draw_rotate_rectangle(img, gt_bev_boxes, color=(0, 255, 0))

        # get 0.1 labels
        label_dir_1 = input_dir + '0.1/' + iter + '/data/'
        boxes3d_pred_1 = load_pred_box3d(label_dir_1, name, score_threshold=0.1)
        gt_bev_boxes_1 = project_label_to_bev_box(boxes3d_pred_1, ground_plane)

        # draw pred bev_box_1
        if len(gt_bev_boxes_1) > 0:
            cv2.line(img, (50, 70), (100, 70), (0, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(img, 'pred_0.1', (110, 70), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 255, 255), 1)
            img = draw_rotate_rectangle(img, gt_bev_boxes_1, color=(0, 255, 255))

        # get 0.1_2 labels
        label_dir_2 = input_dir + '0.1_2/' + iter + '/data/'
        boxes3d_pred_2 = load_pred_box3d(label_dir_2, name, score_threshold=0.1)
        gt_bev_boxes_2 = project_label_to_bev_box(boxes3d_pred_2, ground_plane)

        # draw pred bev_box_2
        if len(gt_bev_boxes_2) > 0:
            cv2.line(img, (50, 90), (100, 90), (255, 0, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(img, 'pred_0.1_2', (110, 90), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255, 0, 255), 1)
            img = draw_rotate_rectangle(img, gt_bev_boxes_2, color=(255,0,255))

        cv2.imwrite(output_dir + name + '.png', img)
        print('Saving image' + output_dir + name + '.png')

def create_video(img_dir):
    output_dir = img_dir + '../'
    # read all image
    image_names = os.listdir(img_dir)
    video_frames = {}

    for name in image_names:
        video_id = name[:2]
        if not video_frames.__contains__(video_id):
            video_frames[video_id] = []
        video_frames[video_id].append(name)

    for (video_id, frames) in video_frames.items():
        frames.sort()
        video_name = output_dir + str(video_id) + '.mp4'
        frame = cv2.imread(img_dir + frames[0])
        height, width, channel = frame.shape
        video = cv2.VideoWriter(video_name, 0, 2, (width, height))
        for image_name in frames:
            image = cv2.imread(img_dir + image_name)
            video.write(image)

        cv2.destroyAllWindows()
        video.release()


def main(_):
    checkpoint_name = 'pyramid_cars_with_aug_dt_5_tracking_corr_pretrained_new'
    default_pipeline_config_path = avod.root_dir() + \
        '/configs/' + checkpoint_name + '.config'
    iter = '7000'
    input_dir = avod.root_dir() + '/data/outputs/'+ checkpoint_name + \
                '/predictions/kitti_native_eval/'

    output_dir = avod.root_dir() + '/../viz/video/corr_tracking_7000/'
    os.makedirs(output_dir, exist_ok=True)
    #
    # Parse pipeline config
    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            default_pipeline_config_path,
            dataset_name='',
            is_training=False)

    # Overwrite data split
    dataset_config.data_split = 'val'
    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    dataset = build_dataset(eval_config, dataset_config)

    # draw_prediction(dataset, input_dir, iter, output_dir)

    create_video(output_dir)


if __name__ == '__main__':
    tf.app.run()