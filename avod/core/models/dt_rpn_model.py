import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from avod.builders import feature_extractor_builder
from avod.core import anchor_encoder
from avod.core import anchor_filter
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core import constants
from avod.core import losses
from avod.core import model
from avod.core import summary_utils
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.datasets.kitti import kitti_aug

from avod.core.corr_layers.correlation import correlation



class DtRpnModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_BEV_INPUT = 'bev_input_pl'
    PL_IMG_INPUT = 'img_input_pl'

    PL_ANCHORS_A = 'anchors_pl_0'

    PL_BEV_ANCHORS_A = 'bev_anchors_pl_0'
    PL_BEV_ANCHORS_NORM_A = 'bev_anchors_norm_pl_0'
    PL_IMG_ANCHORS_A = 'img_anchors_pl_0'
    PL_IMG_ANCHORS_NORM_A = 'img_anchors_norm_pl_0'
    PL_LABEL_ANCHORS_A = 'label_anchors_pl_0'
    PL_LABEL_BOXES_3D_A = 'label_boxes_3d_pl_0'
    PL_LABEL_CLASSES_A = 'label_classes_pl_0'

    PL_ANCHOR_IOUS_A = 'anchor_ious_pl_0'
    PL_ANCHOR_OFFSETS_A = 'anchor_offsets_pl_0'
    PL_ANCHOR_CLASSES_A = 'anchor_classes_pl_0'

    PL_ANCHORS_B = 'anchors_pl_1'

    PL_BEV_ANCHORS_B = 'bev_anchors_pl_1'
    PL_BEV_ANCHORS_NORM_B = 'bev_anchors_norm_pl_1'
    PL_IMG_ANCHORS_B = 'img_anchors_pl_1'
    PL_IMG_ANCHORS_NORM_B = 'img_anchors_norm_pl_1'
    PL_LABEL_ANCHORS_B = 'label_anchors_pl_1'
    PL_LABEL_BOXES_3D_B = 'label_boxes_3d_pl_1'
    PL_LABEL_CLASSES_B = 'label_classes_pl_1'

    PL_ANCHOR_IOUS_B = 'anchor_ious_pl_1'
    PL_ANCHOR_OFFSETS_B = 'anchor_offsets_pl_1'
    PL_ANCHOR_CLASSES_B = 'anchor_classes_pl_1'

    PL_LABEL_CORR_ANCHORS = 'corr_label_anchors_pl'
    PL_LABEL_CORR_BOXES_3D = 'corr_label_boxes_3d_pl'

    # Sample info, including keys for projection to image space
    # (e.g. camera matrix, image index, etc.)
    PL_CALIB_P2 = 'frame_calib_p2'
    PL_IMG_IDX = 'current_img_idx'
    PL_GROUND_PLANE = 'ground_plane'

    ##############################
    # Keys for Predictions
    ##############################
    PRED_ANCHORS = 'rpn_anchors'

    PRED_MB_OBJECTNESS_GT = 'rpn_mb_objectness_gt'
    PRED_MB_OFFSETS_GT = 'rpn_mb_offsets_gt'

    PRED_MB_MASK = 'rpn_mb_mask'
    PRED_MB_OBJECTNESS = 'rpn_mb_objectness'
    PRED_MB_OFFSETS = 'rpn_mb_offsets'

    PRED_TOP_INDICES = 'rpn_top_indices'
    PRED_TOP_ANCHORS = 'rpn_top_anchors'
    PRED_TOP_OBJECTNESS_SOFTMAX = 'rpn_top_objectness_softmax'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_RPN_OBJECTNESS = 'rpn_objectness_loss'
    LOSS_RPN_REGRESSION = 'rpn_regression_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(DtRpnModel, self).__init__(model_config)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test

        self._is_training = (self._train_val_test == 'train')

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        self._img_pixel_size = np.asarray([input_config.img_dims_h,
                                           input_config.img_dims_w])
        self._img_depth = input_config.img_depth

        # Rpn config
        rpn_config = self._config.rpn_config
        self._proposal_roi_crop_size = \
            [rpn_config.rpn_proposal_roi_crop_size] * 2
        self._fusion_method = rpn_config.rpn_fusion_method

        if self._train_val_test in ["train", "val"]:
            self._nms_size = rpn_config.rpn_train_nms_size
        else:
            self._nms_size = rpn_config.rpn_test_nms_size

        self._nms_iou_thresh = rpn_config.rpn_nms_iou_thresh

        # Feature Extractor Nets
        self._bev_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.bev_feature_extractor)
        self._img_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.img_feature_extractor)

        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self.sample_info = dict()

        # Dataset
        self.dataset = dataset
        self.dataset.train_val_test = self._train_val_test
        self._area_extents = self.dataset.kitti_utils.area_extents
        self._bev_extents = self.dataset.kitti_utils.bev_extents
        self._cluster_sizes, _ = self.dataset.get_cluster_info()
        self._anchor_strides = self.dataset.kitti_utils.anchor_strides
        self._anchor_generator = \
            grid_anchor_3d_generator.GridAnchor3dGenerator()

        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._train_on_all_samples = self._config.train_on_all_samples
        self._eval_all_samples = self._config.eval_all_samples
        # Overwrite the dataset's variable with the config
        self.dataset.train_on_all_samples = self._train_on_all_samples

        if self._train_val_test in ["val", "test"]:
            # Disable path-drop, this should already be disabled inside the
            # evaluator, but just in case.
            self._path_drop_probabilities[0] = 1.0
            self._path_drop_probabilities[1] = 1.0

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        # Combine config data
        N = self.dataset.sample_num
        bev_dims = np.append(self._bev_pixel_size, self._bev_depth)
        bev_dims = np.append(N, bev_dims)

        with tf.variable_scope('bev_input'):
            # Placeholder for BEV image input, to be filled in with feed_dict
            bev_input_placeholder_couple = self._add_placeholder(tf.float32, bev_dims,
                                                          self.PL_BEV_INPUT)
            self._bev_input_batches = []
            self._bev_preprocessed = []
            for i in range(N):
                temp_bev_input_batch = tf.expand_dims(bev_input_placeholder_couple[i], axis=0)
                temp_bev_preprocessed = self._bev_feature_extractor.preprocess_input(
                                             temp_bev_input_batch, self._bev_pixel_size)

                self._bev_input_batches.append(temp_bev_input_batch)
                self._bev_preprocessed.append(temp_bev_preprocessed)

            # Summary Images
            bev_summary_images_0 = tf.split(
                bev_input_placeholder_couple[0], self._bev_depth, axis=2)
            tf.summary.image("bev_maps_0", bev_summary_images_0,
                                max_outputs=self._bev_depth)
            bev_summary_images_1 = tf.split(
                bev_input_placeholder_couple[1], self._bev_depth, axis=2)
            tf.summary.image("bev_maps_1", bev_summary_images_1,
                             max_outputs=self._bev_depth)

        with tf.variable_scope('img_input'):
            # Take variable size input images
            img_input_placeholder_couple = self._add_placeholder(
                                                tf.float32,
                                                [N, None, None, self._img_depth],
                                                self.PL_IMG_INPUT)
            self._img_input_batches = []
            self._img_preprocessed = []
            for i in range(N):
                temp_img_input_batches = tf.expand_dims(img_input_placeholder_couple[i], axis=0)
                temp_img_preprocessed = self._img_feature_extractor.preprocess_input(
                                            temp_img_input_batches, self._img_pixel_size)

                self._img_input_batches.append(temp_img_input_batches)
                self._img_preprocessed.append(temp_img_preprocessed)

            # Summary Image
            tf.summary.image("rgb_image_0", self._img_preprocessed[0], max_outputs=2)
            tf.summary.image("rgb_image_1", self._img_preprocessed[1], max_outputs=2)

        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.float32, [None, 6], self.PL_LABEL_ANCHORS_A)
            self._add_placeholder(tf.float32, [None, 7], self.PL_LABEL_BOXES_3D_A)
            self._add_placeholder(tf.float32, [None],    self.PL_LABEL_CLASSES_A)
            self._add_placeholder(tf.float32, [None, 6], self.PL_LABEL_ANCHORS_B)
            self._add_placeholder(tf.float32, [None, 7], self.PL_LABEL_BOXES_3D_B)
            self._add_placeholder(tf.float32, [None,  ], self.PL_LABEL_CLASSES_B)
            self._add_placeholder(tf.float32, [None, 3], self.PL_LABEL_CORR_ANCHORS)
            self._add_placeholder(tf.float32, [None, 3], self.PL_LABEL_CORR_BOXES_3D)

        # Placeholders for anchors
        with tf.variable_scope('pl_anchors'):
            self._add_placeholder(tf.float32, [None, 6], self.PL_ANCHORS_A)
            self._add_placeholder(tf.float32, [None], self.PL_ANCHOR_IOUS_A)
            self._add_placeholder(tf.float32, [None, 6], self.PL_ANCHOR_OFFSETS_A)
            self._add_placeholder(tf.float32, [None], self.PL_ANCHOR_CLASSES_A)
            self._add_placeholder(tf.float32, [None, 6], self.PL_ANCHORS_B)
            self._add_placeholder(tf.float32, [None], self.PL_ANCHOR_IOUS_B)
            self._add_placeholder(tf.float32, [None, 6], self.PL_ANCHOR_OFFSETS_B)
            self._add_placeholder(tf.float32, [None], self.PL_ANCHOR_CLASSES_B)

            with tf.variable_scope('bev_anchor_projections'):
                self._add_placeholder(tf.float32, [None, 4], self.PL_BEV_ANCHORS_A)
                self._bev_anchors_norm_pl_0 = self._add_placeholder(tf.float32, [None, 4],
                                                                  self.PL_BEV_ANCHORS_NORM_A)
                self._add_placeholder(tf.float32, [None, 4], self.PL_BEV_ANCHORS_B)
                self._bev_anchors_norm_pl_1 = self._add_placeholder(tf.float32, [None, 4],
                                                                    self.PL_BEV_ANCHORS_NORM_B)

                self._bev_anchors_norm_pl = [self._bev_anchors_norm_pl_0, self._bev_anchors_norm_pl_1]

            with tf.variable_scope('img_anchor_projections'):
                self._add_placeholder(tf.float32, [None, 4], self.PL_IMG_ANCHORS_A)
                self._img_anchors_norm_pl_0 = self._add_placeholder(tf.float32, [None, 4],
                                                                         self.PL_IMG_ANCHORS_NORM_A)
                self._add_placeholder(tf.float32, [None, 4], self.PL_IMG_ANCHORS_B)
                self._img_anchors_norm_pl_1 = self._add_placeholder(tf.float32, [None, 4],
                                                                           self.PL_IMG_ANCHORS_NORM_B)

                self._img_anchors_norm_pl = [self._img_anchors_norm_pl_0, self._img_anchors_norm_pl_1]

            with tf.variable_scope('sample_info'):
                # the calib matrix shape is (3 x 4)
                self._add_placeholder(tf.float32, [3, 4], self.PL_CALIB_P2)
                self._add_placeholder(tf.int32, shape=[2], name=self.PL_IMG_IDX)
                self._add_placeholder(tf.float32, [2,4], self.PL_GROUND_PLANE)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and
        bottlenecks as member variables.
        """
        length = len(self._bev_preprocessed)
        self.bev_feature_maps = [None]*length
        self.bev_end_points = [None]*length
        self.img_feature_maps = [None]*length
        self.img_end_points = [None]*length

        with tf.variable_scope('feature_extractor') as scope:
            for i in range(length):
                self.bev_feature_maps[i], self.bev_end_points[i] = \
                    self._bev_feature_extractor.build(
                        self._bev_preprocessed[i],
                        self._bev_pixel_size,
                        self._is_training)

                self.img_feature_maps[i], self.img_end_points[i] = \
                    self._img_feature_extractor.build(
                        self._img_preprocessed[i],
                        self._img_pixel_size,
                        self._is_training)

                scope.reuse_variables()

        with tf.variable_scope('bev_bottleneck') as scope:
            self.bev_bottleneck = []
            for bev_feature_map in self.bev_feature_maps:
                temp_bev_bottleneck = slim.conv2d(
                                        bev_feature_map,
                                        1, [1, 1],
                                        scope='bottleneck',
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': self._is_training})

                self.bev_bottleneck.append(temp_bev_bottleneck)
                scope.reuse_variables()

        with tf.variable_scope('img_bottleneck') as scope:
            self.img_bottleneck = []
            for img_feature_map in self.img_feature_maps:
                temp_img_bottleneck = slim.conv2d(
                                        img_feature_map,
                                        1, [1, 1],
                                        scope='bottleneck',
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': self._is_training})

                self.img_bottleneck.append(temp_img_bottleneck)
                scope.reuse_variables()

    def correlation_layer(self):
        corr_config = self._config.layers_config.correlation_config

        with tf.variable_scope('bev_correlation'):
            self.bev_corr_feature_maps = correlation(
                self.bev_feature_maps[0], self.bev_feature_maps[1],
                max_displacement=corr_config.max_displacement,
                padding=corr_config.padding)

        # with tf.variable_scope('img_correlation'):
        #     self.img_corr_feature_maps = correlation(
        #         self.img_feature_maps[0],self.img_feature_maps[1],
        #         max_displacement=corr_config.max_displacement,
        #         padding=corr_config.padding)

        with tf.variable_scope('bev_corr_bottleneck'):
            self.bev_corr_bottleneck = slim.conv2d(
                            self.bev_corr_feature_maps,
                            1, [1, 1],
                            scope='bev_corr_bottleneck',
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': self._is_training})

        # with tf.variable_scope('img_corr_bottleneck'):
        #     self.img_corr_bottleneck = slim.conv2d(self.img_corr_feature_maps,
        #                     1, [1, 1],
        #                     scope='img_corr_bottleneck',
        #                     normalizer_fn=slim.batch_norm,
        #                     normalizer_params={'is_training': self._is_training})


    def build(self):
        SAMPLE_SIZE = self.dataset.sample_num
        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()

        bev_proposal_input = self.bev_bottleneck
        img_proposal_input = self.img_bottleneck

        # compute correlation feature
        self.correlation_layer()

        fusion_mean_div_factor = 2.0

        # If both img and bev probabilites are set to 1.0, don't do
        # path drop.
        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0):
            with tf.variable_scope('rpn_path_drop'):

                random_values = tf.random_uniform(shape=[3],
                                                  minval=0.0,
                                                  maxval=1.0)

                img_mask, bev_mask = self.create_path_drop_masks(
                    self._path_drop_probabilities[0],
                    self._path_drop_probabilities[1],
                    random_values)

                img_proposal_input = [tf.multiply(proposal_input,img_mask)
                                      for proposal_input in img_proposal_input]

                bev_proposal_input = [tf.multiply(proposal_input,bev_mask)
                                      for proposal_input in bev_proposal_input]

                self.img_path_drop_mask = img_mask
                self.bev_path_drop_mask = bev_mask

                # Overwrite the division factor
                fusion_mean_div_factor = img_mask + bev_mask

        with tf.variable_scope('proposal_roi_pooling'):

            with tf.variable_scope('box_indices'):
                def get_box_indices(boxes):
                    proposals_shape = boxes.get_shape().as_list()
                    if any(dim is None for dim in proposals_shape):
                        proposals_shape = tf.shape(boxes)
                    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                    multiplier = tf.expand_dims(
                        tf.range(start=0, limit=proposals_shape[0]), 1)
                    return tf.reshape(ones_mat * multiplier, [-1])

                bev_boxes_norm_batches = [tf.expand_dims(
                    self._bev_anchors_norm_pl[i], axis=0) for i in range(SAMPLE_SIZE)]

                # These should be all 0's since there is only 1 image
                tf_box_indices = [get_box_indices(bev_boxes_norm_batches[i])
                                  for i in range(SAMPLE_SIZE)]

            # Do ROI Pooling on BEV
            bev_proposal_rois = [tf.image.crop_and_resize(
                bev_proposal_input[i],
                self._bev_anchors_norm_pl[i],
                tf_box_indices[i],
                self._proposal_roi_crop_size) for i in range(SAMPLE_SIZE)]
            # Do ROI Pooling on image
            img_proposal_rois = [tf.image.crop_and_resize(
                img_proposal_input[i],
                self._img_anchors_norm_pl[i],
                tf_box_indices[i],
                self._proposal_roi_crop_size) for i in range(SAMPLE_SIZE)]

        with tf.variable_scope('proposal_roi_fusion'):
            rpn_fusion_out = None
            if self._fusion_method == 'mean':
                tf_features_sum = [tf.add(bev_proposal_rois[i], img_proposal_rois[i])
                                   for i in range(SAMPLE_SIZE)]
                rpn_fusion_out = [tf.divide(tf_features_sum[i],fusion_mean_div_factor)
                                  for i in range(SAMPLE_SIZE)]

            elif self._fusion_method == 'concat':
                rpn_fusion_out = [tf.concat([bev_proposal_rois[i], img_proposal_rois[i]],
                                    axis=3) for i in range(SAMPLE_SIZE)]
            else:
                raise ValueError('Invalid fusion method', self._fusion_method)

        # TODO: move this section into an separate AnchorPredictor class
        with tf.variable_scope('anchor_predictor', 'ap', [rpn_fusion_out]) as scope:
            tensor_in = rpn_fusion_out

            # Parse rpn layers config
            layers_config = self._config.layers_config.rpn_config
            l2_weight_decay = layers_config.l2_weight_decay

            if l2_weight_decay > 0:
                weights_regularizer = slim.l2_regularizer(l2_weight_decay)
            else:
                weights_regularizer = None

            cls_fc6 = [None] * 2
            cls_fc6_drop = [None] * 2
            cls_fc7 = [None] * 2
            cls_fc7_drop = [None] * 2
            cls_fc8 = [None] * 2
            objectness = [None] * 2
            reg_fc6 = [None] * 2
            reg_fc6_drop = [None] * 2
            reg_fc7 = [None] * 2
            reg_fc7_drop = [None] * 2
            reg_fc8 = [None] * 2
            offsets = [None] * 2
            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=weights_regularizer):

                for i in range(SAMPLE_SIZE):
                    # Use conv2d instead of fully_connected layers.
                    cls_fc6[i] = slim.conv2d(tensor_in[i],
                                          layers_config.cls_fc6,
                                          self._proposal_roi_crop_size,
                                          padding='VALID',
                                          scope='cls_fc6')

                    cls_fc6_drop[i] = slim.dropout(cls_fc6[i],
                                                layers_config.keep_prob,
                                                is_training=self._is_training,
                                                scope='cls_fc6_drop')

                    cls_fc7[i] = slim.conv2d(cls_fc6_drop[i],
                                          layers_config.cls_fc7,
                                          [1, 1],
                                          scope='cls_fc7')

                    cls_fc7_drop[i] = slim.dropout(cls_fc7[i],
                                                layers_config.keep_prob,
                                                is_training=self._is_training,
                                                scope='cls_fc7_drop')

                    cls_fc8[i] = slim.conv2d(cls_fc7_drop[i],
                                          2,
                                          [1, 1],
                                          activation_fn=None,
                                          scope='cls_fc8')

                    objectness[i] = tf.squeeze(
                                            cls_fc8[i], [1, 2],
                                            name='cls_fc8/squeezed')

                    # Use conv2d instead of fully_connected layers.
                    reg_fc6[i] = slim.conv2d(tensor_in[i],
                                          layers_config.reg_fc6,
                                          self._proposal_roi_crop_size,
                                          padding='VALID',
                                          scope='reg_fc6')

                    reg_fc6_drop[i] = slim.dropout(reg_fc6[i],
                                                layers_config.keep_prob,
                                                is_training=self._is_training,
                                                scope='reg_fc6_drop')

                    reg_fc7[i] = slim.conv2d(reg_fc6_drop[i],
                                          layers_config.reg_fc7,
                                          [1, 1],
                                          scope='reg_fc7')

                    reg_fc7_drop[i] = slim.dropout(reg_fc7[i],
                                                layers_config.keep_prob,
                                                is_training=self._is_training,
                                                scope='reg_fc7_drop')

                    reg_fc8[i] = slim.conv2d(reg_fc7_drop[i],
                                          6,
                                          [1, 1],
                                          activation_fn=None,
                                          scope='reg_fc8')

                    offsets[i] = tf.squeeze(
                                        reg_fc8[i], [1, 2],
                                        name='reg_fc8/squeezed')

                    scope.reuse_variables()

        # Histogram summaries
        with tf.variable_scope('histograms_feature_extractor'):
            with tf.variable_scope('bev_vgg'):
                for i in range(SAMPLE_SIZE):
                    for end_point in self.bev_end_points[i]:
                        tf.summary.histogram(
                            end_point, self.bev_end_points[i][end_point])

            with tf.variable_scope('img_vgg'):
                for i in range(SAMPLE_SIZE):
                    for end_point in self.img_end_points[i]:
                        tf.summary.histogram(
                            end_point, self.img_end_points[i][end_point])

        with tf.variable_scope('histograms_rpn'):
            with tf.variable_scope('anchor_predictor'):
                for i in range(SAMPLE_SIZE):
                    fc_layers = [cls_fc6[i], cls_fc7[i], cls_fc8[i], objectness[i],
                                 reg_fc6[i], reg_fc7[i], reg_fc8[i], offsets[i]]
                    for fc_layer in fc_layers:
                        # fix the name to avoid tf warnings
                        tf.summary.histogram(fc_layer.name.replace(':', '_'),
                                             fc_layer)

        # Return the proposals
        with tf.variable_scope('proposals'):
            anchors = [self.placeholders[self.PL_ANCHORS_A],
                       self.placeholders[self.PL_ANCHORS_B]]

            # Decode anchor regression offsets
            with tf.variable_scope('decoding'):
                regressed_anchors = [anchor_encoder.offset_to_anchor(anchors[i], offsets[i])
                                    for i in range(SAMPLE_SIZE)]

            with tf.variable_scope('bev_projection'):
                bev_proposal_boxes_norm = []
                for i in range(SAMPLE_SIZE):
                    _, bev_proposal_box_norm = anchor_projector.project_to_bev(
                        regressed_anchors[i], self._bev_extents)
                    bev_proposal_boxes_norm.append(bev_proposal_box_norm)

            with tf.variable_scope('softmax'):
                objectness_softmax = [tf.nn.softmax(objectness[i]) for i in range(SAMPLE_SIZE)]

            with tf.variable_scope('nms'):
                objectness_scores = [objectness_softmax[i][:, 1] for i in range(SAMPLE_SIZE)]

                # Do NMS on regressed anchors
                top_indices = [tf.image.non_max_suppression(
                                bev_proposal_boxes_norm[i], objectness_scores[i],
                                max_output_size=self._nms_size,
                                iou_threshold=self._nms_iou_thresh)
                                for i in range(SAMPLE_SIZE)]

                top_anchors = [tf.gather(regressed_anchors[i], top_indices[i])
                               for i in range(SAMPLE_SIZE)]

                top_objectness_softmax = [tf.gather(objectness_scores[i], top_indices[i])
                                          for i in range(SAMPLE_SIZE)]
                # top_offsets = tf.gather(offsets, top_indices)
                # top_objectness = tf.gather(objectness, top_indices)

        # Get mini batch
        all_ious_gt = [self.placeholders[self.PL_ANCHOR_IOUS_A],
                       self.placeholders[self.PL_ANCHOR_IOUS_B]]
        all_offsets_gt = [self.placeholders[self.PL_ANCHOR_OFFSETS_A],
                          self.placeholders[self.PL_ANCHOR_OFFSETS_B]]
        all_classes_gt = [self.placeholders[self.PL_ANCHOR_CLASSES_A],
                          self.placeholders[self.PL_ANCHOR_CLASSES_B]]

        with tf.variable_scope('mini_batch'):
            mini_batch_utils = self.dataset.kitti_utils.mini_batch_utils
            mini_batch_mask = [mini_batch_utils.sample_rpn_mini_batch(all_ious_gt[i])[0]
                               for i in range(SAMPLE_SIZE)]

        # ROI summary images
        rpn_mini_batch_size = \
            self.dataset.kitti_utils.mini_batch_utils.rpn_mini_batch_size
        with tf.variable_scope('bev_rpn_rois'):
            mb_bev_anchors_norm = [tf.boolean_mask(self._bev_anchors_norm_pl[i],
                                                  mini_batch_mask[i])
                                   for i in range(SAMPLE_SIZE)]
            mb_bev_box_indices = [tf.zeros_like(
                tf.boolean_mask(all_classes_gt[i], mini_batch_mask[i]),
                dtype=tf.int32) for i in range(SAMPLE_SIZE)]

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            bev_input_rois = [tf.image.crop_and_resize(
                self._bev_preprocessed[i],
                mb_bev_anchors_norm[i],
                mb_bev_box_indices[i],
                (32, 32))
                for i in range(SAMPLE_SIZE)]

            bev_input_roi_summary_images = [tf.split(
                bev_input_rois[i], self._bev_depth, axis=3)
                for i in range(SAMPLE_SIZE)]
            tf.summary.image('bev_rpn_rois',
                             bev_input_roi_summary_images[0][-1],
                             max_outputs=rpn_mini_batch_size)
            tf.summary.image('bev_rpn_rois',
                             bev_input_roi_summary_images[1][-1],
                             max_outputs=rpn_mini_batch_size)

        with tf.variable_scope('img_rpn_rois'):
            # ROIs on image input
            mb_img_anchors_norm = [tf.boolean_mask(self._img_anchors_norm_pl[i],
                                                  mini_batch_mask[i])
                                   for i in range(SAMPLE_SIZE)]
            mb_img_box_indices = [tf.zeros_like(
                tf.boolean_mask(all_classes_gt[i], mini_batch_mask[i]),
                dtype=tf.int32)for i in range(SAMPLE_SIZE)]

            # Do test ROI pooling on mini batch
            img_input_rois = [tf.image.crop_and_resize(
                self._img_preprocessed[i],
                mb_img_anchors_norm[i],
                mb_img_box_indices[i],
                (32, 32)) for i in range(SAMPLE_SIZE)]

            tf.summary.image('img_rpn_rois',
                             img_input_rois[0],
                             max_outputs=rpn_mini_batch_size)
            tf.summary.image('img_rpn_rois',
                             img_input_rois[1],
                             max_outputs=rpn_mini_batch_size)

        # Ground Truth Tensors
        with tf.variable_scope('one_hot_classes'):

            # Anchor classification ground truth
            # Object / Not Object
            min_pos_iou = \
                self.dataset.kitti_utils.mini_batch_utils.rpn_pos_iou_range[0]

            objectness_classes_gt = [tf.cast(
                tf.greater_equal(all_ious_gt[i], min_pos_iou),
                dtype=tf.int32) for i in range(SAMPLE_SIZE)]

            objectness_gt = [tf.one_hot(
                objectness_classes_gt[i], depth=2,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=self._config.label_smoothing_epsilon)
                for i in range(SAMPLE_SIZE)]

        # Mask predictions for mini batch
        with tf.variable_scope('prediction_mini_batch'):
            objectness_masked = [tf.boolean_mask(objectness[i], mini_batch_mask[i])
                                 for i in range(SAMPLE_SIZE)]
            offsets_masked = [tf.boolean_mask(offsets[i], mini_batch_mask[i])
                              for i in range(SAMPLE_SIZE)]

        with tf.variable_scope('ground_truth_mini_batch'):
            objectness_gt_masked = [tf.boolean_mask(objectness_gt[i], mini_batch_mask[i])
                                    for i in range(SAMPLE_SIZE)]
            offsets_gt_masked = [tf.boolean_mask(all_offsets_gt[i],mini_batch_mask[i])
                                 for i in range(SAMPLE_SIZE)]

        # Specify the tensors to evaluate
        predictions = dict()

        # Temporary predictions for debugging
        # predictions['anchor_ious'] = anchor_ious
        # predictions['anchor_offsets'] = all_offsets_gt

        if self._train_val_test in ['train', 'val']:
            # All anchors
            predictions[self.PRED_ANCHORS] = anchors

            # Mini-batch masks
            predictions[self.PRED_MB_MASK] = mini_batch_mask
            # Mini-batch predictions
            predictions[self.PRED_MB_OBJECTNESS] = objectness_masked
            predictions[self.PRED_MB_OFFSETS] = offsets_masked

            # Mini batch ground truth
            predictions[self.PRED_MB_OFFSETS_GT] = offsets_gt_masked
            predictions[self.PRED_MB_OBJECTNESS_GT] = objectness_gt_masked

            # Proposals after nms
            predictions[self.PRED_TOP_INDICES] = top_indices
            predictions[self.PRED_TOP_ANCHORS] = top_anchors
            predictions[
                self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax

        else:
            # self._train_val_test == 'test'
            predictions[self.PRED_TOP_ANCHORS] = top_anchors
            predictions[self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax

        return predictions

    def create_feed_dict(self, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """

        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None:
                raise ValueError('sample_index should be None. Do not load '
                                 'particular samples during train or val')

            # During training/validation, we need a valid sample
            # with anchor info for loss calculation
            couple_sample = None
            anchors_info = []

            valid_sample = False
            while not valid_sample:
                if self._train_val_test == "train":
                    # Get the a random sample from the remaining epoch
                    couple_samples = self.dataset.next_batch(batch_size=1)

                else:  # self._train_val_test == "val"
                    # Load samples in order for validation
                    couple_samples = self.dataset.next_batch(batch_size=1,
                                                      shuffle=False)

                # Only handle one sample at a time for now
                couple_sample = couple_samples[0]
                anchors_info = couple_sample.get(constants.KEY_ANCHORS_INFO)

                # When training, if the mini batch is empty, go to the next
                # sample. Otherwise carry on with found the valid sample.
                # For validation, even if 'anchors_info' is empty, keep the
                # sample (this will help penalize false positives.)
                # We will substitue the necessary info with zeros later on.
                # Note: Training/validating all samples can be switched off.
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)

                anchors_valid = (len(anchors_info[0]) > 0) and (len(anchors_info[1]) > 0)
                if anchors_valid or train_cond or eval_cond:
                    valid_sample = True
        else:
            # For testing, any sample should work
            if sample_index is not None:
                couple_samples = self.dataset.load_samples([sample_index])
            else:
                couple_samples = self.dataset.next_batch(batch_size=1, shuffle=False)

            # Only handle one sample at a time for now
            couple_sample = couple_samples[0]
            anchors_info = couple_sample.get(constants.KEY_ANCHORS_INFO)

        sample_name = couple_sample.get(constants.KEY_SAMPLE_NAME)
        sample_augs = couple_sample.get(constants.KEY_SAMPLE_AUGS)

        # Get ground truth data
        label_anchors = couple_sample.get(constants.KEY_LABEL_ANCHORS)
        label_classes = couple_sample.get(constants.KEY_LABEL_CLASSES)
        # We only need orientation from box_3d
        label_boxes_3d = couple_sample.get(constants.KEY_LABEL_BOXES_3D)

        # Network input data
        image_input = couple_sample.get(constants.KEY_IMAGE_INPUT)
        bev_input = couple_sample.get(constants.KEY_BEV_INPUT)

        # correlation gt offsets
        label_corr_boxes_3d = couple_sample.get(constants.KEY_LABEL_CORR_BOXES_3D)
        label_corr_anchors = couple_sample.get(constants.KEY_LABEL_CORR_ANCHORS)

        # Image shape (h, w)
        image_shape = [[image.shape[0], image.shape[1]] for image in image_input]

        ground_plane = couple_sample.get(constants.KEY_GROUND_PLANE)


        stereo_calib_p2 = couple_sample.get(constants.KEY_STEREO_CALIB_P2)

        # Fill the placeholders for anchor information
        self._fill_anchor_pl_inputs(anchors_info=anchors_info,
                                    ground_plane=ground_plane,
                                    image_shape=image_shape,
                                    stereo_calib_p2=stereo_calib_p2,
                                    sample_name=sample_name,
                                    sample_augs=sample_augs)

        # this is a list to match the explicit shape for the placeholder
        # Fill in the rest
        self._placeholder_inputs[self.PL_BEV_INPUT] = bev_input
        self._placeholder_inputs[self.PL_IMG_INPUT] = image_input

        self._placeholder_inputs[self.PL_LABEL_ANCHORS_A] = label_anchors[0][:, :-1]
        self._placeholder_inputs[self.PL_LABEL_BOXES_3D_A] = label_boxes_3d[0][:, :-1]
        self._placeholder_inputs[self.PL_LABEL_CLASSES_A] = label_classes[0]

        self._placeholder_inputs[self.PL_LABEL_ANCHORS_B] = label_anchors[1][:, :-1]
        self._placeholder_inputs[self.PL_LABEL_BOXES_3D_B] = label_boxes_3d[1][:, :-1]
        self._placeholder_inputs[self.PL_LABEL_CLASSES_B] = label_classes[1]

        # x,z,ry
        self._placeholder_inputs[self.PL_LABEL_CORR_BOXES_3D] = label_corr_boxes_3d[:, [0, 2, 6]]
        self._placeholder_inputs[self.PL_LABEL_CORR_ANCHORS] = label_corr_anchors[:, :3]

        # Sample Info
        # img_idx is a list to match the placeholder shape
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name[0]), int(sample_name[1])]
        self._placeholder_inputs[self.PL_CALIB_P2] = stereo_calib_p2
        self._placeholder_inputs[self.PL_GROUND_PLANE] = ground_plane

        # Temporary sample info for debugging
        self.sample_info.clear()
        self.sample_info['sample_name'] = sample_name
        self.sample_info['rpn_mini_batch'] = anchors_info

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict

    def _fill_anchor_pl_inputs(self,
                               anchors_info,
                               ground_plane,
                               image_shape,
                               stereo_calib_p2,
                               sample_name,
                               sample_augs):
        """
        Fills anchor placeholder inputs with corresponding data

        Args:
            anchors_info: anchor info from mini_batch_utils
            ground_plane: ground plane coefficients
            image_shape: image shape (h, w), used for projecting anchors
            sample_name: name of the sample, e.g. "000001"
            sample_augs: list of sample augmentations
        """
        self._bev_anchors_norm = []
        self._img_anchors_norm = []
        self._placeholder_inputs["PL_ANCHORS"] = []
        self._placeholder_inputs["PL_ANCHOR_IOUS"] = []
        self._placeholder_inputs["PL_ANCHOR_OFFSETS"] = []
        self._placeholder_inputs["PL_ANCHOR_CLASSES"] = []
        self._placeholder_inputs["PL_BEV_ANCHORS"] = []
        self._placeholder_inputs["PL_IMG_ANCHORS"] = []

        # unpack anchors_info
        for i in range(len(sample_name)):
            # Lists for merging anchors info
            all_anchor_boxes_3d = []
            anchors_ious = []
            anchor_offsets = []
            anchor_classes = []

            # Create anchors for each class
            if len(self.dataset.classes) > 1:
                for class_idx in range(len(self.dataset.classes)):
                    # Generate anchors for all classes
                    grid_anchor_boxes_3d = self._anchor_generator.generate(
                        area_3d=self._area_extents,
                        anchor_3d_sizes=self._cluster_sizes[class_idx],
                        anchor_stride=self._anchor_strides[class_idx],
                        ground_plane=ground_plane[i])
                    all_anchor_boxes_3d.append(grid_anchor_boxes_3d)
                all_anchor_boxes_3d = np.concatenate(all_anchor_boxes_3d)
            else:
                # Don't loop for a single class
                class_idx = 0
                grid_anchor_boxes_3d = self._anchor_generator.generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=self._cluster_sizes[class_idx],
                    anchor_stride=self._anchor_strides[class_idx],
                    ground_plane=ground_plane[i])
                all_anchor_boxes_3d = grid_anchor_boxes_3d

            # Filter empty anchors
            # Skip if anchors_info is []
            sample_has_labels = True
            if self._train_val_test in ['train', 'val']:
                # Read in anchor info during training / validation
                if anchors_info[i]:
                    anchor_indices, anchors_ious, anchor_offsets, \
                        anchor_classes = anchors_info[i]

                    anchor_boxes_3d_to_use = all_anchor_boxes_3d[anchor_indices]
                else:
                    train_cond = (self._train_val_test == "train" and
                                  self._train_on_all_samples)
                    eval_cond = (self._train_val_test == "val" and
                                 self._eval_all_samples)
                    if train_cond or eval_cond:
                        sample_has_labels = False
            else:
                sample_has_labels = False

            if not sample_has_labels:
                # During testing, or validation with no anchor info, manually
                # filter empty anchors
                # TODO: share voxel_grid_2d with BEV generation if possible
                voxel_grid_2d = \
                    self.dataset.kitti_utils.create_sliced_voxel_grid_2d(
                        sample_name[i], self.dataset.bev_source,
                        image_shape=image_shape[i])

                # Convert to anchors and filter
                anchors_to_use = box_3d_encoder.box_3d_to_anchor(
                    all_anchor_boxes_3d)
                empty_filter = anchor_filter.get_empty_anchor_filter_2d(
                    anchors_to_use, voxel_grid_2d, density_threshold=1)

                anchor_boxes_3d_to_use = all_anchor_boxes_3d[empty_filter]

            # Convert lists to ndarrays
            anchor_boxes_3d_to_use = np.asarray(anchor_boxes_3d_to_use)
            anchors_ious = np.asarray(anchors_ious)
            anchor_offsets = np.asarray(anchor_offsets)
            anchor_classes = np.asarray(anchor_classes)

            # Flip anchors and centroid x offsets for augmented samples
            if kitti_aug.AUG_FLIPPING in sample_augs:
                anchor_boxes_3d_to_use = kitti_aug.flip_boxes_3d(
                    anchor_boxes_3d_to_use, flip_ry=False)
                if anchors_info[i]:
                    anchor_offsets[:, 0] = -anchor_offsets[:, 0]

            # Convert to anchors
            anchors_to_use = box_3d_encoder.box_3d_to_anchor(anchor_boxes_3d_to_use)
            num_anchors = len(anchors_to_use)

            # Project anchors into bev
            bev_anchors, bev_anchors_norm = anchor_projector.project_to_bev(
                anchors_to_use, self._bev_extents)

            # Project box_3d anchors into image space
            img_anchors, img_anchors_norm = \
                anchor_projector.project_to_image_space(
                    anchors_to_use, stereo_calib_p2, image_shape[i])

            # Reorder into [y1, x1, y2, x2] for tf.crop_and_resize op
            self._bev_anchors_norm.append(bev_anchors_norm[:, [1, 0, 3, 2]])
            self._img_anchors_norm.append(img_anchors_norm[:, [1, 0, 3, 2]])

            # Fill in placeholder inputs
            if i == 0:
                self._placeholder_inputs[self.PL_ANCHORS_A] = anchors_to_use

                # If we are in train/validation mode, and the anchor infos
                # are not empty, store them. Checking for just anchors_ious
                # to be non-empty should be enough.
                if self._train_val_test in ['train', 'val'] and \
                        len(anchors_ious) > 0:
                    self._placeholder_inputs[self.PL_ANCHOR_IOUS_A] = anchors_ious
                    self._placeholder_inputs[self.PL_ANCHOR_OFFSETS_A] = anchor_offsets
                    self._placeholder_inputs[self.PL_ANCHOR_CLASSES_A] = anchor_classes

                # During test, or val when there is no anchor info
                elif self._train_val_test in ['test'] or \
                        len(anchors_ious) == 0:
                    # During testing, or validation with no gt, fill these in with 0s
                    self._placeholder_inputs[self.PL_ANCHOR_IOUS_A] = np.zeros(num_anchors)
                    self._placeholder_inputs[self.PL_ANCHOR_OFFSETS_A] = np.zeros([num_anchors, 6])
                    self._placeholder_inputs[self.PL_ANCHOR_CLASSES_A] = np.zeros(num_anchors)
                else:
                    raise ValueError('Got run mode {}, and non-empty anchor info'.
                                        format(self._train_val_test))

                self._placeholder_inputs[self.PL_BEV_ANCHORS_A] = bev_anchors
                self._placeholder_inputs[self.PL_IMG_ANCHORS_A] = img_anchors
            else:
                self._placeholder_inputs[self.PL_ANCHORS_B] = anchors_to_use

                # If we are in train/validation mode, and the anchor infos
                # are not empty, store them. Checking for just anchors_ious
                # to be non-empty should be enough.
                if self._train_val_test in ['train', 'val'] and \
                        len(anchors_ious) > 0:
                    self._placeholder_inputs[self.PL_ANCHOR_IOUS_B] = anchors_ious
                    self._placeholder_inputs[self.PL_ANCHOR_OFFSETS_B] = anchor_offsets
                    self._placeholder_inputs[self.PL_ANCHOR_CLASSES_B] = anchor_classes

                # During test, or val when there is no anchor info
                elif self._train_val_test in ['test'] or \
                        len(anchors_ious) == 0:
                    # During testing, or validation with no gt, fill these in with 0s
                    self._placeholder_inputs[self.PL_ANCHOR_IOUS_B] = np.zeros(num_anchors)
                    self._placeholder_inputs[self.PL_ANCHOR_OFFSETS_B] = np.zeros([num_anchors, 6])
                    self._placeholder_inputs[self.PL_ANCHOR_CLASSES_B] = np.zeros(num_anchors)
                else:
                    raise ValueError('Got run mode {}, and non-empty anchor info'.
                                     format(self._train_val_test))

                self._placeholder_inputs[self.PL_BEV_ANCHORS_B] = bev_anchors
                self._placeholder_inputs[self.PL_IMG_ANCHORS_B] = img_anchors

        self._placeholder_inputs[self.PL_BEV_ANCHORS_NORM_A] = self._bev_anchors_norm[0]
        self._placeholder_inputs[self.PL_BEV_ANCHORS_NORM_B] = self._bev_anchors_norm[1]
        self._placeholder_inputs[self.PL_IMG_ANCHORS_NORM_A] = self._img_anchors_norm[0]
        self._placeholder_inputs[self.PL_IMG_ANCHORS_NORM_B] = self._img_anchors_norm[1]

    def loss(self, prediction_dict):
        SAMPLE_SIZE = 2
        # these should include mini-batch values only
        objectness_gt = prediction_dict[self.PRED_MB_OBJECTNESS_GT]
        offsets_gt = prediction_dict[self.PRED_MB_OFFSETS_GT]

        # Predictions
        with tf.variable_scope('rpn_prediction_mini_batch'):
            objectness = prediction_dict[self.PRED_MB_OBJECTNESS]
            offsets = prediction_dict[self.PRED_MB_OFFSETS]

        with tf.variable_scope('rpn_losses'):
            with tf.variable_scope('objectness'):
                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                objectness_loss = [cls_loss(objectness[i],
                                           objectness_gt[i],
                                           weight=cls_loss_weight)
                                   for i in range(SAMPLE_SIZE)]

                with tf.variable_scope('obj_norm'):
                    # normalize by the number of anchor mini-batches
                    objectness_loss = [objectness_loss[i] / tf.cast(
                                        tf.shape(objectness_gt[i])[0], dtype=tf.float32)
                                       for i in range(SAMPLE_SIZE)]

                    tf.summary.scalar('objectness', objectness_loss[0])
                    tf.summary.scalar('objectness', objectness_loss[1])

            with tf.variable_scope('regression'):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                anchorwise_localization_loss = [reg_loss(offsets[i],
                                                        offsets_gt[i],
                                                        weight=reg_loss_weight)
                                                for i in range(SAMPLE_SIZE)]
                masked_localization_loss = \
                    [anchorwise_localization_loss[i] * objectness_gt[i][:, 1]
                    for i in range(SAMPLE_SIZE)]

                localization_loss = [tf.reduce_sum(masked_localization_loss[i])
                                     for i in range(SAMPLE_SIZE)]

                with tf.variable_scope('reg_norm'):
                    # normalize by the number of positive objects
                    num_positives = [tf.reduce_sum(objectness_gt[i][:, 1])
                                     for i in range(SAMPLE_SIZE)]
                    # Assert the condition `num_positives > 0`
                    for i in range(SAMPLE_SIZE):
                        with tf.control_dependencies(
                                [tf.assert_positive(num_positives[i])]):
                            localization_loss[i] = localization_loss[i] / num_positives[i]
                            tf.summary.scalar('regression', localization_loss[i])

            objectness_loss = tf.reduce_sum(objectness_loss)
            localization_loss = tf.reduce_sum(localization_loss)
            with tf.variable_scope('total_loss'):
                total_loss = objectness_loss + localization_loss

        loss_dict = {
            self.LOSS_RPN_OBJECTNESS: objectness_loss,
            self.LOSS_RPN_REGRESSION: localization_loss,
        }

        return loss_dict, total_loss

    def create_path_drop_masks(self,
                               p_img,
                               p_bev,
                               random_values):
        """Determines global path drop decision based on given probabilities.

        Args:
            p_img: A tensor of float32, probability of keeping image branch
            p_bev: A tensor of float32, probability of keeping bev branch
            random_values: A tensor of float32 of shape [3], the results
                of coin flips, values should range from 0.0 - 1.0.

        Returns:
            final_img_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
            final_bev_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
        """

        def keep_branch(): return tf.constant(1.0)

        def kill_branch(): return tf.constant(0.0)

        # The logic works as follows:
        # We have flipped 3 coins, first determines the chance of keeping
        # the image branch, second determines keeping bev branch, the third
        # makes the final decision in the case where both branches were killed
        # off, otherwise the initial img and bev chances are kept.

        img_chances = tf.case([(tf.less(random_values[0], p_img),
                                keep_branch)], default=kill_branch)

        bev_chances = tf.case([(tf.less(random_values[1], p_bev),
                                keep_branch)], default=kill_branch)

        # Decision to determine whether both branches were killed off
        third_flip = tf.logical_or(tf.cast(img_chances, dtype=tf.bool),
                                   tf.cast(bev_chances, dtype=tf.bool))
        third_flip = tf.cast(third_flip, dtype=tf.float32)

        # Make a second choice, for the third case
        # Here we use a 50/50 chance to keep either image or bev
        # If its greater than 0.5, keep the image
        img_second_flip = tf.case([(tf.greater(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)
        # If its less than or equal to 0.5, keep bev
        bev_second_flip = tf.case([(tf.less_equal(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)

        # Use lambda since this returns another condition and it needs to
        # be callable
        final_img_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: img_chances)],
                                 default=lambda: img_second_flip)

        final_bev_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: bev_chances)],
                                 default=lambda: bev_second_flip)

        return final_img_mask, final_bev_mask