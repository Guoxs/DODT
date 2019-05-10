"""Dataset utils for preparing data for the network."""

import itertools
import fnmatch
import os
import random

import numpy as np
import cv2

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import tracking_utils

from avod.core import box_3d_encoder
from avod.core import constants
from avod.datasets.kitti import kitti_aug
from avod.datasets.kitti.kitti_tracking_utils import KittiTrackingUtils, Oxts


class Sample:
    def __init__(self, name, augs):
        self.name = name
        self.augs = augs


class KittiTrackingDataset:
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

        self.is_final_train = self.config.is_final_train

        # 2 image a samples
        self.sample_num = 2

        # stride for couple data, default is 1
        self.data_stride = self.config.data_stride

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

        # sample mini-batch for superpameters searching
        if not self.is_final_train:
            mini_ids = random.sample(list(range(len(self.sample_list))), 200)
            self.sample_list = self.sample_list[mini_ids]

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
            anchor_info = self.kitti_utils.get_tracking_anchors_info(
                                            self.classes_name,
                                            self.kitti_utils.anchor_strides,
                                            sample_name)
            anchors_info.append(anchor_info)
        return anchors_info

    def get_video_frames(self, video_id):
        set_file = self.dataset_dir + '/' + self.data_split + '.txt'
        data_list = []
        with open(set_file, 'r') as f:
            sample_names = f.read().split('\n\n')
            items = sample_names[video_id]
            items = items.split('\n')
            if items[-1] == '':
                items = items[:-1]
            for line in items:
                names = line.split('/')
                video_id = int(names[0])
                frame_id = int(names[1])
                data_name = str(video_id).zfill(2) + str(frame_id).zfill(4)
                data_list.append(data_name)
        return data_list


    # generate sample couple
    def generate_sample_couple(self):
        '''split frames into batches, each batch contains 'seq_len' (default is 2)
        continuous frames, with sliding step of 'stride' (default is 1)

        return:
            dataList: batch list, each item in list likes this: ['000000', '000001'],
                      first substring is video id, and the rest are four continuous frames id.
                      the last frame will be duplicated if necessary: ['000153', '000153']
        '''
        def extract_id(name):
            video_id = int(name.split('/')[0])
            frame_id = int(name.split('/')[1])
            return str(video_id).zfill(2) + str(frame_id).zfill(4)

        def split_video_ids(ids, stride, data_list):
            ids = list(map(extract_id, ids))
            for i in range(len(ids)):
                cur = ids[i]
                if i+stride < len(ids):
                    next = ids[i+stride]
                else:
                    next = ids[-1]

                data_list.append([cur, next])
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
                frame_num = int(item[-1].split('/')[1])
                assert len(item) == frame_num+1, print('Frame number match failed!')
                if self.data_split == 'test':
                    data_list = split_video_ids(item, self.data_stride, data_list)
                elif video_id in self.video_train_id:
                    if self.data_split in ['train', 'trainval']:
                        data_list = split_video_ids(item, self.data_stride, data_list)
                else:
                    if self.data_split in ['val', 'trainval']:
                        data_list = split_video_ids(item, self.data_stride, data_list)
        return data_list


    def coordinate_transform(self, sample_names):
        '''get translation vector and rotation matrix, in order to represent previous frame
            in current frame's coordinate.

        input:
            sample_names:  ['010001', '010002']

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
        point_cloud[-1] = pc_next.T
        return point_cloud


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
            labels[-1] = label_next
        return labels

    def inv_label_transform(self, t, ry, sample_names):
        trans, matrix, delta = self.coordinate_transform(sample_names)
        ori_t = t @ np.linalg.inv(matrix) - trans
        ori_delta = ry - delta
        return ori_t, ori_delta

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

            # Only read labels if they exist
            if self.has_labels:
                # Read mini batch first to see if it is empty
                anchors_info = self.get_anchors_info(sample_names)
                not_empty = (len(anchors_info[0]) > 0) and (len(anchors_info[1]) > 0)
                if not not_empty and self.train_val_test == 'train' \
                        and (not self.train_on_all_samples):
                    anchors_info = []
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

                anchors_info = [[],[]]

                label_anchors = [np.zeros((1, 7)), np.zeros((1, 7))]
                label_boxes_3d = [np.zeros((1, 8)), np.zeros((1, 8))]
                label_classes = [np.zeros(1), np.zeros(1)]

            # Load image (BGR -> RGB)
            cv_bgr_image = [cv2.imread(self.get_rgb_image_path(name)) for name in sample_names]
            rgb_image = [image[..., :: -1] for image in cv_bgr_image]
            image_shape = [img.shape[0:2] for img in rgb_image]
            # resize to the same shape
            if image_shape[0] != image_shape[1]:
                rgb_image[-1] = cv2.resize(rgb_image[-1], image_shape[0])
                image_shape[1] = image_shape[0]
            image_input = rgb_image

            # Get ground plane
            ground_plane = [tracking_utils.get_road_plane(name, self.planes_dir)
                            for name in sample_names]

            # Get calibration
            assert sample_names[0][:2] == sample_names[1][:2], print("sample couple from different video!")
            stereo_calib_p2 = calib_utils.read_tracking_calibration(self.calib_dir,
                                                           int(sample_names[0][:2])).p2

            # load raw lidar data
            raw_point_cloud = [self.kitti_utils.get_raw_point_cloud(
                self.bev_source, sample_names[i]) for i in range(len(sample_names))]

            # transform second point_cloud frame to first point_cloud frame coordinate system
            point_cloud = self.point_cloud_transform(raw_point_cloud, sample_names)

            # convert transfered lidar to camera view
            point_cloud = [self.kitti_utils.transfer_lidar_to_camera_view(
                self.bev_source, sample_names[i], point_cloud[i], image_shape[i]
            ) for i in range(len(sample_names))]

            # transform second frame label to first frame label coordinate system
            if obj_labels is not None:
                obj_labels = self.label_transform(obj_labels, sample_names)

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

            if obj_labels is not None:
                label_boxes_3d = []
                label_anchors = []
                label_classes = []
                for i in range(len(obj_labels)):
                    label_box_3d = np.asarray(
                        [box_3d_encoder.tracking_object_label_to_box_3d(obj_label)
                         for obj_label in obj_labels[i]])

                    label_class = [
                        self.kitti_utils.class_str_to_index(obj_label.type)
                        for obj_label in obj_labels[i]]
                    label_class = np.asarray(label_class, dtype=np.int32)

                    # Return empty anchors_info if no ground truth after filtering
                    if len(label_box_3d) == 0:
                        anchors_info[i] = []
                        if self.train_on_all_samples:
                            # If training without any positive labels, we cannot
                            # set these to zeros, because later on the offset calc
                            # uses log on these anchors. So setting any arbitrary
                            # number here that does not break the offset calculation
                            # should work, since the negative samples won't be
                            # regressed in any case.
                            dummy_anchors = [[-1000, -1000, -1000, 1, 1, 1]]
                            label_anchor = np.asarray(dummy_anchors)
                            dummy_boxes = [[-1000, -1000, -1000, 1, 1, 1, 0]]
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

            # Create BEV maps
            bev_images = [self.kitti_utils.create_bev_maps(
                point_cloud[i], ground_plane[i]) for i in range(len(sample_names))]

            height_maps = [bev_image.get('height_maps') for bev_image in bev_images]
            density_map = [bev_image.get('density_map') for bev_image in bev_images]
            bev_input = [np.dstack((*height_maps[i], density_map[i]))
                         for i in range(len(bev_images))]

            #bev_input = [np.sum(bev, axis=2)for bev in bev_input]
            bev_input = np.asarray(bev_input)
            #bev_input = np.expand_dims(bev_input, axis=2)

            # calculate correlation offsets
            label_corr_boxes_3d = self.calculate_corr_offsets(label_boxes_3d)
            label_corr_anchors = self.calculate_corr_offsets(label_anchors)

            # align anchors_info
            aligned_anchors_info = []
            anchors_info_mask = []
            if len(anchors_info[0]) > 0 and len(anchors_info[1]) > 0:
                anchors_info_mask, _ = self.list_align([anchors_info[0][0],
                                    anchors_info[1][0]], return_mask=True)

                for (item1, item2) in zip(anchors_info[0], anchors_info[1]):
                    aligned_anchors_info.append(
                        self.list_align([item1, item2]))

            label_mask, _ = self.list_align(label_boxes_3d, return_mask=True)
            # transpose point_cloud for data align
            point_cloud = [point_cloud[0].T, point_cloud[1].T]
            point_cloud_mask, point_cloud = self.list_align(point_cloud, return_mask=True)

            sample_dict = {
                constants.KEY_LABEL_BOXES_3D: self.list_align(label_boxes_3d),
                constants.KEY_LABEL_ANCHORS: self.list_align(label_anchors),
                constants.KEY_LABEL_CLASSES: self.list_align(label_classes),
                constants.KEY_LABEL_MASK: label_mask,

                constants.KEY_LABEL_CORR_BOXES_3D: label_corr_boxes_3d,
                constants.KEY_LABEL_CORR_ANCHORS: label_corr_anchors,

                constants.KEY_IMAGE_INPUT: np.asarray(image_input),
                constants.KEY_BEV_INPUT: bev_input,

                constants.KEY_ANCHORS_INFO: aligned_anchors_info,
                constants.KEY_ANCHORS_INFO_MASK: anchors_info_mask,

                constants.KEY_POINT_CLOUD: point_cloud,
                constants.KEY_POINT_CLOUD_MASK: point_cloud_mask,
                constants.KEY_GROUND_PLANE: np.asarray(ground_plane),
                constants.KEY_STEREO_CALIB_P2: stereo_calib_p2,

                constants.KEY_SAMPLE_NAME: sample_names,
                constants.KEY_SAMPLE_AUGS: sample.augs
            }
            sample_dicts.append(sample_dict)

        return sample_dicts

    def list_align(self, list, return_mask=False):
        len1 = list[0].shape[0]
        len2 = list[1].shape[0]
        mask = np.zeros((len1+len2,), dtype = np.int32)
        mask[len1:] = 1
        out = np.concatenate(list, axis=0)
        if return_mask:
            return mask, out
        else:
            return out

    def get_from_idx(self, data, idx):
        assert len(data.shape) == 2, print('shape unmatch!')
        mask = data[:, 0]
        n_idx = (mask == idx)
        return data[n_idx]

    def calculate_corr_offsets(self, labels):
        labels_1 = labels[0]
        labels_2 = labels[1]
        assert len(labels_1) > 0, print('invaild label!')
        assert len(labels_2) > 0, print('invaild label!')
        corr_offsets = np.zeros_like(labels_1)
        for i in range(len(labels_1)):
            label = labels_1[i]
            if label.all() == 0:
                continue
            obj_id = label[-1]
            match_flag = False
            for j in range(len(labels_2)):
                d_label = labels_2[j]
                d_obj_id = d_label[-1]
                if obj_id == d_obj_id:
                    match_flag = True
                    corr_offsets[i] = d_label - label
                    corr_offsets[i][-1] = obj_id

            # object does not exist in frame 2
            if not match_flag:
                #corr_offsets[i] = - label
                corr_offsets[i][-1] = obj_id
        return corr_offsets



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