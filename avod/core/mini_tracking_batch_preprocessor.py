# import cv2
import numpy as np
import os

from PIL import Image

from wavedata.tools.obj_detection import tracking_utils
from wavedata.tools.obj_detection import evaluation

from avod.core import box_3d_encoder, anchor_projector
from avod.core import anchor_encoder
from avod.core import anchor_filter

from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.core.mini_batch_preprocessor import MiniBatchPreprocessor


class MiniTrackingBatchPreprocessor(MiniBatchPreprocessor):
    def __init__(self,
                 dataset,
                 mini_batch_dir,
                 anchor_strides,
                 density_threshold,
                 neg_iou_3d_range,
                 pos_iou_3d_range):
        """Preprocesses anchors and saves info to files for RPN training

        Args:
            dataset: Dataset object
            mini_batch_dir: directory to save the info
            anchor_strides: anchor strides for generating anchors (per class)
            density_threshold: minimum number of points required to keep an
                anchor
            neg_iou_3d_range: 3D iou range for an anchor to be negative
            pos_iou_3d_range: 3D iou range for an anchor to be positive
        """
        super(MiniTrackingBatchPreprocessor, self).__init__(
            dataset,
            mini_batch_dir,
            anchor_strides,
            density_threshold,
            neg_iou_3d_range,
            pos_iou_3d_range
        )

    def preprocess(self, indices):
        """Preprocesses anchor info and saves info to files

        Args:
            indices (int array): sample indices to process.
                If None, processes all samples
        """
        # Get anchor stride for class
        anchor_strides = self._anchor_strides

        dataset = self._dataset
        dataset_utils = self._dataset.kitti_utils
        classes_name = dataset.classes_name

        # Make folder if it doesn't exist yet
        output_dir = self.mini_batch_utils.get_file_path(classes_name,
                                                         anchor_strides,
                                                         sample_name=None)
        os.makedirs(output_dir, exist_ok=True)

        # Get clusters for class
        all_clusters_sizes, _ = dataset.get_cluster_info()

        anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

        # Load indices of data_split
        all_samples = dataset.sample_list

        if indices is None:
            indices = np.arange(len(all_samples))
        num_samples = len(indices)

        # For each image in the dataset, save info on the anchors
        for sample_idx in indices:
            # Get image name for given cluster
            sample_couple_name = all_samples[sample_idx].name
            # Get ground truth and filter based on difficulty
            ground_truth_lists = [tracking_utils.read_labels(dataset.label_dir, name)
                                  for name in sample_couple_name]

            # do coordinate transform for label
            ground_truth_lists = self._dataset.label_transform(ground_truth_lists, sample_couple_name)

            for i in range(len(sample_couple_name)):
                sample_name = sample_couple_name[i]

                # Check for existing files and skip to the next
                if self._check_for_existing(classes_name, anchor_strides,
                                            sample_name):
                    print("{} / {}: Sample already preprocessed".format(
                        sample_name, num_samples, sample_name))
                    continue

                # Get ground truth and filter based on difficulty
                ground_truth_list = ground_truth_lists[i]

                # Filter objects to dataset classes
                filtered_gt_list = dataset_utils.filter_labels(ground_truth_list)
                filtered_gt_list = np.asarray(filtered_gt_list)

                # Filtering by class has no valid ground truth, skip this image
                if len(filtered_gt_list) == 0:
                    print("{} / {} No {}s for sample {} "
                          "(Ground Truth Filter)".format(
                                sample_name, num_samples,
                              classes_name, sample_name))

                    # Output an empty file and move on to the next image.
                    self._save_to_file(classes_name, anchor_strides, sample_name)
                    continue

                # Get ground plane
                ground_plane = tracking_utils.get_road_plane(sample_name,
                                                        dataset.planes_dir)

                image = Image.open(dataset.get_rgb_image_path(sample_name))
                image_shape = [image.size[1], image.size[0]]

                # Generate sliced 2D voxel grid for filtering
                vx_grid_2d = dataset_utils.create_sliced_voxel_grid_2d(
                    sample_name,
                    source=dataset.bev_source,
                    image_shape=image_shape)

                # List for merging all anchors
                all_anchor_boxes_3d = []

                # Create anchors for each class
                for class_idx in range(len(dataset.classes)):
                    # Generate anchors for all classes
                    grid_anchor_boxes_3d = anchor_generator.generate(
                        area_3d=self._area_extents,
                        anchor_3d_sizes=all_clusters_sizes[class_idx],
                        anchor_stride=self._anchor_strides[class_idx],
                        ground_plane=ground_plane)

                    all_anchor_boxes_3d.extend(grid_anchor_boxes_3d)

                # Filter empty anchors
                all_anchor_boxes_3d = np.asarray(all_anchor_boxes_3d)
                anchors = box_3d_encoder.box_3d_to_anchor(all_anchor_boxes_3d)
                empty_anchor_filter = anchor_filter.get_empty_anchor_filter_2d(
                    anchors, vx_grid_2d, self._density_threshold)

                # Calculate anchor info
                anchors_info = self._calculate_anchors_info(
                    all_anchor_boxes_3d, empty_anchor_filter, filtered_gt_list)

                anchor_ious = anchors_info[:, self.mini_batch_utils.col_ious]

                valid_iou_indices = np.where(anchor_ious > 0.0)[0]

                print("{} / {}:"
                      "{:>6} anchors, "
                      "{:>6} iou > 0.0, "
                      "for {:>3} {}(s) for sample {}".format(
                          sample_name, num_samples,
                          len(anchors_info),
                          len(valid_iou_indices),
                          len(filtered_gt_list), classes_name, sample_name
                      ))

                # Save anchors info
                self._save_to_file(classes_name, anchor_strides,
                                   sample_name, anchors_info)

