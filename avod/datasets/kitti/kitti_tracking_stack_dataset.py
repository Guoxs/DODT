"""Dataset utils for preparing data for the network."""

import itertools
import fnmatch
import os

import numpy as np
import cv2

from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.core.mini_tracking_batch_preprocessor import MiniTrackingBatchPreprocessor
from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import tracking_utils

from avod.core import box_3d_encoder, box_8c_encoder, box_4c_encoder
from avod.core import constants
from avod.datasets.kitti import kitti_aug
from avod.datasets.kitti.kitti_tracking_utils import KittiTrackingUtils, Oxts
from avod.datasets.kitti.label_offset import cal_label_offsets


class Sample:
    def __init__(self, name, augs):
        self.name = name
        self.augs = augs


class KittiTrackingStackDataset:
    def __init__(self, dataset_config):
        """
        Initializes directories, and loads the sample list

        Args:
            dataset_config: KittiDatasetConfig
                name: unique name for the dataset
                data_split: "train", "val", "test", "trainval"
                data_split_dir: must be specified for custom "training"
                dataset_dir: Kitti dataset dir if not in default location
                classes: relevant classes
                num_clusters: number of k-means clusters to separate for
                    each class
        """
        # Parse config
        self.config = dataset_config

        self.name = self.config.name
        self.data_split = self.config.data_split
        self.dataset_dir = os.path.expanduser(self.config.dataset_dir)
        data_split_dir = self.config.data_split_dir

        self.has_labels = self.config.has_labels
        self.cluster_split = self.config.cluster_split

        self.classes = list(self.config.classes)
        self.num_classes = len(self.classes)
        self.num_clusters = np.asarray(self.config.num_clusters)

        self.bev_source = self.config.bev_source
        self.aug_list = self.config.aug_list

        # 3 image a samples
        self.sample_num = 2
        self.stride = self.config.data_stride

        # stride for couple data, default is 1
        # self.temporal_batch = self.config.data_stride

        # video id for training split
        self.video_train_id = self.config.video_train_id

        # Determines the network mode. This is initialized to 'train' but
        # is overwritten inside the model based on the mode.
        self.train_val_test = 'train'
        # Determines if training includes all samples, including the ones
        # without anchor_info. This is initialized to False, but is overwritten
        # via the config inside the model.
        self.train_on_all_samples = False

        self._set_up_classes_name()

        # Check that paths and split are valid
        self._check_dataset_dir()

        # Get all files and folders in dataset directory
        all_files = os.listdir(self.dataset_dir)

        # Get possible data splits from txt files in dataset folder
        possible_splits = []
        for file_name in all_files:
            if fnmatch.fnmatch(file_name, '*.txt'):
                possible_splits.append(os.path.splitext(file_name)[0])
        # This directory contains a readme.txt file, remove it from the list
        if 'readme' in possible_splits:
            possible_splits.remove('readme')

        if self.data_split not in possible_splits:
            raise ValueError("Invalid data split: {}, possible_splits: {}"
                             .format(self.data_split, possible_splits))

        # Check data_split_dir
        # Get possible data split dirs from folder names in dataset folder
        possible_split_dirs = []
        for folder_name in all_files:
            if os.path.isdir(self.dataset_dir + '/' + folder_name):
                possible_split_dirs.append(folder_name)
        if data_split_dir in possible_split_dirs:
            split_dir = self.dataset_dir + '/' + data_split_dir
            self._data_split_dir = split_dir
        else:
            raise ValueError(
                "Invalid data split dir: {}, possible dirs".format(
                    data_split_dir, possible_split_dirs))

        # Batch pointers
        self._index_in_epoch = 0
        self.epochs_completed = 0

        self._cam_idx = 2

        # Initialize the sample list
        loaded_sample_couple_names = self.generate_sample_couple()

        # Augment the sample list
        aug_sample_list = []

        # Loop through augmentation lengths e.g.
        # 0: []
        # 1: ['flip'], ['pca_jitter']
        # 2: ['flip', 'pca_jitter']
        for aug_idx in range(len(self.aug_list) + 1):
            # Get all combinations
            augmentations = list(itertools.combinations(self.aug_list,
                                                        aug_idx))
            for augmentation in augmentations:
                for sample_name in loaded_sample_couple_names:
                    aug_sample_list.append(Sample(sample_name, augmentation))

        self.sample_list = np.asarray(aug_sample_list)
        self.num_samples = len(self.sample_list)

        self._set_up_directories()

        # Setup utils object
        self.kitti_utils = KittiTrackingUtils(self)

    # Paths
    @property
    def rgb_image_dir(self):
        return self.image_dir

    @property
    def sample_names(self):
        # This is a property since the sample list gets shuffled for training
        return np.asarray(
            [sample.name for sample in self.sample_list])

    @property
    def bev_image_dir(self):
        raise NotImplementedError("BEV images not saved to disk yet!")

    def _check_dataset_dir(self):
        """Checks that dataset directory exists in the file system

        Raises:
            FileNotFoundError: if the dataset folder is missing
        """
        # Check that dataset path is valid
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError('Dataset path does not exist: {}'
                                    .format(self.dataset_dir))

    def _set_up_directories(self):
        """Sets up data directories."""
        # Setup Directories
        self.image_dir = self._data_split_dir + '/image_' + str(self._cam_idx)
        self.calib_dir = self._data_split_dir + '/calib'
        self.planes_dir = self._data_split_dir + '/planes'
        self.velo_dir = self._data_split_dir + '/velodyne'
        self.oxts_dir = self._data_split_dir + '/oxts'

        # Labels are always in the training folder
        self.label_dir = self.dataset_dir + \
            '/training/label_' + str(self._cam_idx)

    def _set_up_classes_name(self):
        # Unique identifier for multiple classes
        if self.num_classes > 1:
            if self.classes == ['Pedestrian', 'Cyclist']:
                self.classes_name = 'People'
            if self.classes == ['Car', 'Van']:
                self.classes_name = 'Car'
            elif self.classes == ['Car', 'Pedestrian', 'Cyclist']:
                self.classes_name = 'All'
            else:
                raise NotImplementedError('Need new unique identifier for '
                                          'multiple classes')
        else:
            self.classes_name = self.classes[0]

    # Get sample paths
    def get_rgb_image_path(self, sample_name):
        assert len(sample_name) == 6, print('Sample name incorrect!')
        video_id = int(sample_name[:2])
        frame_id = int(sample_name[2:])
        return self.rgb_image_dir + '/%04d/%06d.png' %(video_id, frame_id)

    def get_velodyne_path(self, sample_name):
        assert len(sample_name) == 6, print('Sample name incorrect!')
        video_id = int(sample_name[:2])
        frame_id = int(sample_name[2:])
        return self.velo_dir + '/%04d/%06d.bin' % (video_id, frame_id)

    def get_bev_sample_path(self, sample_name):
        assert len(sample_name) == 6, print('Sample name incorrect!')
        video_id = int(sample_name[:2])
        frame_id = int(sample_name[2:])
        return self.bev_image_dir + '/%04d/%06d.png' % (video_id, frame_id)

    def get_oxts(self, sample_name):
        assert len(sample_name) == 6, print('Sample name incorrect!')
        video_id = int(sample_name[:2])
        frame_id = int(sample_name[2:])
        oxts_file = open(self.oxts_dir + '/%04d.txt' % video_id)
        lines = [line.rstrip() for line in oxts_file.readlines()]
        oxts_line = lines[frame_id]
        oxts_file.close()
        return Oxts(oxts_line)

    # Cluster info
    def get_cluster_info(self):
        return self.kitti_utils.clusters, self.kitti_utils.std_devs

    def get_anchors_info(self, sample_names):
        anchors_info = []
        for sample_name in sample_names:
            anchor_info = self.kitti_utils.get_anchors_info(
                                            self.classes_name,
                                            self.kitti_utils.anchor_strides,
                                            sample_name)
            anchors_info.append(anchor_info)
        return anchors_info

    def get_integrated_anchors_info(self, point_cloud, ground_truth_list, ground_plane):
        mini_batch_utils = self.kitti_utils.mini_batch_utils
        integrated_anchors_info = \
            mini_batch_utils.preprocess_rpn_mini_batch_single(
            point_cloud, ground_truth_list, ground_plane)

        integrated_indices = np.asarray(integrated_anchors_info[:, 0], dtype=np.int32)
        integrated_ious = np.asarray(integrated_anchors_info[:, 1], dtype=np.float32)
        integrated_offsets = np.asarray(integrated_anchors_info[:, 2:8], dtype=np.float32)
        integrated_classes = np.asarray(integrated_anchors_info[:, 8], dtype=np.float32)

        return integrated_indices, integrated_ious, \
               integrated_offsets, integrated_classes


    # generate sample couple
    def generate_sample_couple(self):
        '''split frames into batches, each batch contains 'seq_len' continuous frames

        return:
            dataList: batch list, each item in list likes this: ['000000',..., '000003'],
                      first two substring is video id, and the rest are four continuous frames id.
                      the last frame will be duplicated if necessary: ['000153', ..., '000153']
        '''
        def extract_id(name):
            video_id = int(name.split('/')[0])
            frame_id = int(name.split('/')[1])
            return str(video_id).zfill(2) + str(frame_id).zfill(4)

        def split_train_video_ids(ids, num, data_list):
            ids = list(map(extract_id, ids))
            for i in range(len(ids)):
                temp = []
                for j in range(num):
                    if i + j < len(ids):
                        next = ids[i+j]
                    else:
                        next = ids[-1]
                    temp.append(next)
                data_list.append([temp[0], temp[-1]])
            return data_list

        def split_val_test_video_ids(ids, num, data_list):
            ids = list(map(extract_id, ids))
            for i in range(0, len(ids), num-1):
                temp = []
                temp_idx = 0
                while temp_idx < num:
                    if i + temp_idx < len(ids):
                        next = ids[i + temp_idx]
                    else:
                        next = ids[-1]
                    temp.append(next)
                    temp_idx += 1

                data_list.append([temp[0], temp[-1]])
            return data_list

        set_file = self.dataset_dir + '/' + self.data_split + '.txt'
        data_list = []
        with open(set_file, 'r') as f:
            sample_names = f.read().split('\n\n')
            for item in sample_names:
                item = item.split('\n')
                if item[-1] == '':
                    item = item[:-1]
                video_id = int(item[0].split('/')[0])
                # frame_num = int(item[-1].split('/')[1])
                # assert len(item) == frame_num+1, print('Frame number match failed!')
                if self.data_split == 'test':
                    data_list = split_val_test_video_ids(item, self.stride, data_list)
                elif self.data_split == 'trainval':
                    data_list = split_train_video_ids(item, self.stride, data_list)
                elif video_id in self.video_train_id:
                    if self.data_split == 'train':
                        data_list = split_train_video_ids(item, self.stride, data_list)
                else:
                    if self.data_split == 'val':
                        data_list = split_val_test_video_ids(item, self.stride, data_list)
        return data_list


    def coordinate_transform(self, sample_names):
        '''get translation vector and rotation matrix, in order to represent previous frame
            in current frame's coordinate.

        input:
            sample_names:  ['010001','010003']

        output:
            distance:   translation vector      1 x 3
            R:          rotation matrix         3 x 3
        '''
        oxts_cur = self.get_oxts(sample_names[0])
        oxts_next = self.get_oxts(sample_names[1])
        distance = oxts_cur.displacement(oxts_next)
        delta = oxts_cur.get_delta(oxts_next, theta='yaw')
        Rz = oxts_cur.get_rotate_matrix(oxts_next, 'z')
        Rx = oxts_cur.get_rotate_matrix(oxts_next, 'x')
        Ry = oxts_cur.get_rotate_matrix(oxts_next, 'y')
        matrix = Rz @ Rx @ Ry
        return distance, matrix, delta

    def point_cloud_transform(self, point_cloud, sample_names):
        '''
        transform second point_cloud frame to first point_cloud frame coordinate system
        :param point_cloud:     [array(3, N), array(3, N)]
        :param sample_names:    ['010001', '010002']
        :return:                [array(3, N), array(3, N)]
        '''
        trans, matrix, _ = self.coordinate_transform(sample_names)
        pc_next = point_cloud[-1].T
        pc_next[:,:3] = (pc_next[:, :3] + trans) @ matrix
        return pc_next.T


    def label_transform(self, labels, sample_names):
        '''
        transform second frame labels to first frame labels coordinate system
        :param labels:          [array(TrackingLabel object), array(TrackingLabel object)]
        :param sample_names:    ['010001', '010002']
        :return:                [array(TrackingLabel object), array(TrackingLabel object)]
        '''
        def cal_new_t(label_obj, calib, trans, matrix):
            from wavedata.tools.obj_detection import obj_utils
            box3d = obj_utils.compute_box_corners_3d(label_obj).T  #[8,3]
            # transfer to velo coord
            box3d = calib.project_rect_to_velo(box3d)
            # do rotate
            box3d = (box3d + trans) @ matrix
            # back to cam coord
            box3d = calib.project_velo_to_rect(box3d)
            # cal box center
            new_t = np.mean(box3d, axis=0)
            # cal center bottom
            new_t[1] += label_obj.h / 2.0
            return new_t

        trans, matrix, delta = self.coordinate_transform(sample_names)
        # transfer trans to camera coord
        calib = self.kitti_utils.get_calib(self.bev_source, sample_names[-1])
        label_next = labels[-1]
        if len(label_next) != 0:
            for i in range(len(label_next)):
                label_next[i].t = cal_new_t(label_next[i],calib, trans, matrix)
                label_next[i].ry += delta
        return label_next

    def recovery_t(self, label_obj, calib, trans, matrix):
        from wavedata.tools.obj_detection import obj_utils
        box3d = obj_utils.compute_box_corners_3d(label_obj).T  # [8,3]
        # transfer to velo coord
        box3d = calib.project_rect_to_velo(box3d)
        # do rotate
        inv_matrix = np.linalg.inv(matrix)
        box3d = box3d @ inv_matrix - trans
        # back to cam coord
        box3d = calib.project_velo_to_rect(box3d)
        # cal box center
        origin_t = np.mean(box3d, axis=0)
        # cal center bottom
        origin_t[1] += label_obj.h / 2.0
        return origin_t

    def label_inverse_transform(self, labels, sample_names):


        trans, matrix, delta = self.coordinate_transform(sample_names)
        # transfer trans to camera coord
        calib = self.kitti_utils.get_calib(self.bev_source, sample_names[-1])
        label_trans = labels[-1]
        if len(label_trans) != 0:
            for i in range(len(label_trans)):
                label_trans[i].t = self.recovery_t(label_trans[i],calib, trans, matrix)
                label_trans[i].ry -= delta
        return label_trans

    def merge_labels(self, gt_labels):
        assert len(gt_labels) > 0, print('Empty gt_labels!')
        first_label = gt_labels[0]
        # change list to map
        label_dict = dict()
        for obj in first_label:
            label_dict[obj.object_id] = [obj]
        # add labels from other frames
        if len(gt_labels) > 1:
            for i in range(1, len(gt_labels)):
                temp_label = gt_labels[i]
                for obj in temp_label:
                    obj_id = obj.object_id
                    if obj_id in label_dict.keys():
                        label_dict[obj_id].append(obj)
                    else:
                        label_dict[obj_id] = [obj]
        # merge labels
        merged_labels = []
        for obj_list in label_dict.values():
            if len(obj_list) == 1:
                merged_labels.append(obj_list[0])
            else:
                base_labels = obj_list[0]
                # merge boxes2d and boxes3d
                boxes2d = []
                boxes3d = []
                for obj in obj_list:
                    boxes2d.append([obj.x1, obj.x2, obj.y1, obj.y2])
                    boxes3d.append([obj.t[0], obj.t[1], obj.t[2], obj.l, obj.w, obj.h, obj.ry])

                # calculate merged boxes2d
                boxes2d = np.array(boxes2d)
                base_labels.x1 = np.min(boxes2d[:, 0])
                base_labels.x2 = np.max(boxes2d[:, 1])
                base_labels.y1 = np.min(boxes2d[:, 2])
                base_labels.y2 = np.max(boxes2d[:, 3])

                # calculated merged boxes3d
                boxes3d = np.array(boxes3d, dtype=np.float32)
                ground_plane = np.asarray([0.0, -1.0, 0.0, 1.65], dtype=np.float32)
                # [x1, x2, x3, x4, z1, z2, z3, z4, h1, h2]
                boxes3d_4c = [box_4c_encoder.np_box_3d_to_box_4c(box3d, ground_plane)
                              for box3d in boxes3d]
                boxes3d_4c = np.array(boxes3d_4c, dtype=np.float32)
                min_x = np.min(boxes3d_4c[:,:4])
                max_x = np.max(boxes3d_4c[:,:4])
                min_z = np.min(boxes3d_4c[:,4:8])
                max_z = np.max(boxes3d_4c[:,4:8])
                h1 = np.mean(boxes3d_4c[:, 8])
                h2 = np.mean(boxes3d_4c[:, 9])
                new_boxes3d_4c = [max_x, max_x, min_x, min_x, max_z, min_z, min_z, max_z, h1, h2]
                new_boxes3d_4c = np.array(new_boxes3d_4c)
                new_boxes3d = box_4c_encoder.np_box_4c_to_box_3d(new_boxes3d_4c, ground_plane)

                base_labels.t = (new_boxes3d[0], new_boxes3d[1], new_boxes3d[2])
                base_labels.l = new_boxes3d[3]
                base_labels.w = new_boxes3d[4]
                base_labels.h = new_boxes3d[5]
                base_labels.ry = new_boxes3d[6]

                # append to merged_labels
                merged_labels.append(base_labels)
        return merged_labels


    def create_all_sample_names(self, sample_names):
        '''
        :param sample_names: ['010001', '010004']
        :return: ['010001', '010002', '010003', '010004']
        '''
        all_sample_names = [sample_names[0]]
        video_id = sample_names[0][:2]
        frame_id_1 = int(sample_names[0][2:])
        frame_id_2 = int(sample_names[1][2:])

        temp_id = frame_id_1
        while(temp_id < frame_id_2):
            temp_id += 1
            temp_name = video_id + str(temp_id).zfill(4)
            all_sample_names.append(temp_name)

        return all_sample_names

    def load_samples(self, indices):
        """ Loads input-output data for a set of samples. Should only be
                    called when a particular sample dict is required. Otherwise,
                    samples should be provided by the next_batch function

        Args:
            indices: A list of sample indices from the dataset.sample_list
                to be loaded

        Return:
            samples: a list of data sample dicts
        """
        sample_dicts = []
        for sample_idx in indices:
            sample = self.sample_list[sample_idx]
            sample_names = sample.name

            # create all sample names in a batch for integrated point cloud
            all_sample_names = self.create_all_sample_names(sample_names)

            # compute integrated info
            assert self.sample_num == len(sample_names)
            # Only read labels if they exist
            if self.has_labels:
                # Read mini batch first to see if it is empty
                anchors_info = self.get_anchors_info(sample_names)
                not_emptys = [len(info) > 0 for info in anchors_info]
                if not sum(not_emptys) and self.train_val_test == 'train' \
                        and (not self.train_on_all_samples):
                    empty_sample_dict = {
                        constants.KEY_SAMPLE_NAME: sample_names,
                        constants.KEY_ANCHORS_INFO: anchors_info
                    }
                    return [empty_sample_dict]

                obj_labels = [tracking_utils.read_labels(self.label_dir, name)
                              for name in sample_names]

                # Only use objects that match dataset classes
                for i in range(len(obj_labels)):
                    obj_labels[i] = self.kitti_utils.filter_labels(obj_labels[i])

            else:
                obj_labels = None
                label_anchors = [np.zeros((1, 7)) for _ in range(self.sample_num)]
                label_boxes_3d = [np.zeros((1, 8)) for _ in range(self.sample_num)]
                label_classes = [np.zeros(1) for _ in range(self.sample_num)]

                integrated_anchors_info = [[]]
                integrated_label_anchor = np.zeros((1,7))
                integrated_label_box_3d = np.zeros((1,8))
                corr_offsets = np.zeros((1, 4))

            # Load image (BGR -> RGB)
            cv_bgr_image = [cv2.imread(self.get_rgb_image_path(name))
                            for name in sample_names]
            rgb_image = [image[..., :: -1] for image in cv_bgr_image]
            # resize to the same shape
            image_shape = rgb_image[0].shape[0:2]
            for i in range(len(rgb_image)):
                if rgb_image[i].shape[0:2] != image_shape:
                    rgb_image[i] = cv2.resize(rgb_image[i], image_shape)
            image_input = rgb_image
            image_shape = [image_shape for _ in range(self.sample_num)]

            # Get ground plane
            ground_plane = [tracking_utils.get_road_plane(name, self.planes_dir)
                            for name in sample_names]

            # Get calibration
            same_videos = [sample_name[:2] for sample_name in sample_names]
            assert len(set(same_videos)) == 1, \
                print("sample couple should from the same video!")
            stereo_calib_p2 = calib_utils.read_tracking_calibration(
                self.calib_dir, int(sample_names[0][:2])).p2

            # load raw lidar data
            all_raw_point_cloud = [self.kitti_utils.get_raw_point_cloud(
                self.bev_source, all_sample_names[i]) for i in range(len(all_sample_names))]

            # transform other point_cloud frames to first point_cloud frame coordinate system
            point_cloud = [all_raw_point_cloud[0]]
            if len(all_sample_names) > 1:
                for i in range(1, len(all_sample_names)):
                    new_pc = self.point_cloud_transform([point_cloud[0], all_raw_point_cloud[i]],
                                                        [all_sample_names[0], all_sample_names[i]])
                    point_cloud.append(new_pc)

            # convert transfered lidar to camera view
            point_cloud = [self.kitti_utils.transfer_lidar_to_camera_view(
                            self.bev_source, all_sample_names[i], point_cloud[i], image_shape[0])
                            for i in range(len(all_sample_names))]

            # transform second frame label to first frame label coordinate system
            if obj_labels is not None:
                new_labels = [obj_labels[0]]
                for i in range(1, self.sample_num):
                    new_label = self.label_transform([new_labels[0], obj_labels[i]],
                                                     [sample_names[0], sample_names[i]])
                    new_labels.append(new_label)
                obj_labels = new_labels

                # Augmentation (Flipping)
                if kitti_aug.AUG_FLIPPING in sample.augs:
                    image_input = [kitti_aug.flip_image(image) for image in image_input]
                    point_cloud = [kitti_aug.flip_point_cloud(pc) for pc in point_cloud]

                    if obj_labels is not None:
                        for i in range(len(obj_labels)):
                            if obj_labels[i].shape[0] != 0:
                                obj_labels[i] = [kitti_aug.flip_label_in_3d_only(obj)
                                                 for obj in obj_labels[i]]

                    ground_plane = [kitti_aug.flip_ground_plane(plane) for plane in ground_plane]
                    stereo_calib_p2 = kitti_aug.flip_stereo_calib_p2(
                        stereo_calib_p2, image_shape[0])

                # Augmentation (Image Jitter)
                if kitti_aug.AUG_PCA_JITTER in sample.augs:
                    for i in range(len(image_input)):
                        image_input[i][:, :, 0:3] = kitti_aug.apply_pca_jitter(
                            image_input[i][:, :, 0:3])

            # compute integrated point_cloud
            integrated_point_cloud = np.concatenate(point_cloud, axis=1)

            if obj_labels is not None:
                label_boxes_3d = []
                label_anchors = []
                label_classes = []
                for i in range(len(obj_labels)):
                    label_box_3d = np.asarray(
                        [box_3d_encoder.tracking_object_label_to_box_3d(obj_label)
                         for obj_label in obj_labels[i]])

                    label_class = [self.kitti_utils.class_str_to_index(obj_label.type)
                                        for obj_label in obj_labels[i]]
                    label_class = np.asarray(label_class, dtype=np.int32)

                    # Return empty anchors_info if no ground truth after filtering
                    if len(label_box_3d) == 0:
                        if self.train_on_all_samples:
                            # If training without any positive labels, we cannot
                            # set these to zeros, because later on the offset calc
                            # uses log on these anchors. So setting any arbitrary
                            # number here that does not break the offset calculation
                            # should work, since the negative samples won't be
                            # regressed in any case.
                            dummy_anchors = [[-1000, -1000, -1000, 1, 1, 1, 0]]
                            label_anchor = np.asarray(dummy_anchors)
                            dummy_boxes = [[-1000, -1000, -1000, 1, 1, 1, 0, 0]]
                            label_box_3d = np.asarray(dummy_boxes)
                        else:
                            label_anchor = np.zeros((1, 7))
                            label_box_3d = np.zeros((1, 8))
                        label_class = np.zeros(1)
                    else:
                        label_anchor = box_3d_encoder.tracking_box_3d_to_anchor(
                            label_box_3d, ortho_rotate=True)

                    label_boxes_3d.append(label_box_3d)
                    label_anchors.append(label_anchor)
                    label_classes.append(label_class)

                # compute integrated gt_labels
                integrated_obj_labels = self.merge_labels(obj_labels)

                integrated_label_box_3d = np.asarray(
                    [box_3d_encoder.tracking_object_label_to_box_3d(obj_label)
                     for obj_label in integrated_obj_labels])

                integrated_label_class = [self.kitti_utils.class_str_to_index(obj_label.type)
                                          for obj_label in integrated_obj_labels]
                integrated_label_class = np.asarray(integrated_label_class, dtype=np.int32)

                # caluate correlation offsets
                if len(integrated_label_box_3d) == 0:
                    if self.train_on_all_samples:
                        dummy_offsets = [[-1000, -1000, -1000, 0]]
                        corr_offsets = np.asarray(dummy_offsets)
                    else:
                        corr_offsets = np.zeros((1, 4))
                else:
                    corr_offsets = cal_label_offsets(label_boxes_3d[0],
                                                     label_boxes_3d[1])

                if len(integrated_label_box_3d) == 0:
                    integrated_anchors_info = []
                    if self.train_on_all_samples:
                        dummy_anchors = [[-1000, -1000, -1000, 1, 1, 1, 0]]
                        integrated_label_anchor = np.asarray(dummy_anchors)
                        dummy_boxes = [[-1000, -1000, -1000, 1, 1, 1, 0, 0]]
                        integrated_label_box_3d = np.asarray(dummy_boxes)
                    else:
                        integrated_label_anchor = np.zeros((1, 7))
                        integrated_label_box_3d = np.zeros((1, 8))
                    integrated_label_class = np.zeros(1)
                else:
                    integrated_label_anchor = box_3d_encoder.tracking_box_3d_to_anchor(
                        integrated_label_box_3d, ortho_rotate=True)

                    # calculate integrated anchors_info
                    integrated_anchors_info = self.get_integrated_anchors_info(
                        integrated_point_cloud, integrated_obj_labels, ground_plane[0])

            # Create BEV maps
            dual_point_cloud = [point_cloud[0], point_cloud[-1]]
            bev_images = [self.kitti_utils.create_bev_maps(
                dual_point_cloud[i], ground_plane[i]) for i in range(len(sample_names))]

            height_maps = [bev_image.get('height_maps') for bev_image in bev_images]
            density_map = [bev_image.get('density_map') for bev_image in bev_images]
            bev_input = [np.dstack((*height_maps[i], density_map[i]))
                         for i in range(len(bev_images))]

            # bev map for correlation
            single_bev_maps = [np.mean(input[:, :, :-1], axis=-1, keepdims=True)
                               for input in bev_input]

            # Create integrated BEV maps
            integrated_bev_map = self.kitti_utils.create_bev_maps(
                integrated_point_cloud, ground_plane[0])
            integrated_height_map = integrated_bev_map.get('height_maps')
            integrated_density_map = integrated_bev_map.get('density_map')
            integrated_bev_input = np.dstack((*integrated_height_map,
                                         integrated_density_map))

            point_cloud = [pc.T for pc in point_cloud]
            integrated_point_cloud = integrated_point_cloud.T

            # align lists to one
            label_mask, label_boxes_3d = self.list_align(label_boxes_3d)
            _, label_anchors = self.list_align(label_anchors)
            _, label_classes = self.list_align(label_classes)

            point_cloud_mask, point_cloud = self.list_align(point_cloud)

            sample_dict = {
                constants.KEY_LABEL_BOXES_3D: label_boxes_3d,
                constants.KEY_LABEL_ANCHORS: label_anchors,
                constants.KEY_LABEL_CLASSES: label_classes,
                constants.KEY_LABEL_MASK: label_mask,

                constants.KEY_IMAGE_INPUT: np.asarray(image_input),
                constants.KEY_BEV_INPUT: np.asarray(bev_input),

                constants.KEY_POINT_CLOUD: point_cloud,
                constants.KEY_GROUND_PLANE: np.asarray(ground_plane),
                constants.KEY_STEREO_CALIB_P2: stereo_calib_p2,

                constants.KEY_INTEGRATED_BEV_INPUT: integrated_bev_input,
                constants.KEY_INTEGRATED_POINT_CLOUD: integrated_point_cloud,
                constants.KEY_INTEGRATED_ANCHORS_INFO: integrated_anchors_info,
                constants.KEY_INTEGRATED_LABEL_ANCHOR: integrated_label_anchor,
                constants.KEY_INTEGRATED_LABEL_BOX_3D: integrated_label_box_3d,
                constants.KEY_INTEGRATED_LABEL_CLASS: integrated_label_class,

                # for correlation
                constants.KEY_SINGLE_BEV_MAPS: single_bev_maps,
                constants.KEY_CORR_OFFSETS: corr_offsets,

                constants.KEY_SAMPLE_NAME: sample_names,
                constants.KEY_SAMPLE_AUGS: sample.augs
            }
            sample_dicts.append(sample_dict)

        return sample_dicts

    def align_anchors(self, anchors):
        indices, ious, offsets, classes = [], [], [], []
        for anchor in anchors:
            if anchor != []:
                indices.append(anchor[0])
                ious.append(anchor[1])
                offsets.append(anchor[2])
                classes.append(anchor[3])
            else:
                indices.append([])
                ious.append([])
                offsets.append([])
                classes.append([])

        aligned_mask, aligned_indices = self.list_align(indices)
        _, aligned_ious = self.list_align(ious)
        _, aligned_offsets = self.list_align(offsets)
        _, aligned_classes = self.list_align(classes)
        aligned_anchors = (aligned_indices, aligned_ious,
                           aligned_offsets, aligned_classes)
        return aligned_mask, aligned_anchors


    def list_align(self, list):
        lens = [len(item) for item in list]
        masks = [np.ones(lens[i]) * i for i in range(len(lens))]
        final_mask = np.concatenate(masks, axis=0).astype(np.int32)
        out = np.concatenate(list, axis=0)
        return final_mask, out

    def _shuffle_samples(self):
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        self.sample_list = self.sample_list[perm]

    def next_batch(self, batch_size, shuffle=True):
        """
        Retrieve the next `batch_size` samples from this data set.

        Args:
            batch_size: number of samples in the batch
            shuffle: whether to shuffle the indices after an epoch is completed

        Returns:
            list of dictionaries containing sample information
        """

        # Create empty set of samples
        samples_in_batch = []

        start = self._index_in_epoch
        # Shuffle only for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self._shuffle_samples()

        # Go to the next epoch
        if start + batch_size >= self.num_samples:

            # Finished epoch
            self.epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self.num_samples - start

            # Append those samples to the current batch
            samples_in_batch.extend(
                self.load_samples(np.arange(start, self.num_samples)))

            # Shuffle the data
            if shuffle:
                self._shuffle_samples()

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            # Append the rest of the batch
            samples_in_batch.extend(self.load_samples(np.arange(start, end)))

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            # Append the samples in the range to the batch
            samples_in_batch.extend(self.load_samples(np.arange(start, end)))

        return samples_in_batch