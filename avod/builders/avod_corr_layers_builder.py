import tensorflow as tf
from tensorflow.contrib import slim

from avod.core.avod_fc_layers import avod_fc_layer_utils


def build(layers_config, input_rois, is_training):
    """Builds second stage fully connected layers

    Args:
        layers_config: Configuration object
        bev_rois: List of input corr ROI feature maps
        box_rep: Box representation (e.g. 'box_3d', 'box_8c', etc.)
        is_training (bool): Whether the network is training or evaluating

    Returns:
        corr_out: correlation feature output
    """

    with tf.variable_scope('corr_predictor'):
        fc_layers_type = layers_config.WhichOneof('fc_layers')

        if fc_layers_type == 'basic_fc_layers':
            corr_layers_config = layers_config.basic_fc_layers

            corr_out = basic_corr_layers(
                    corr_layers_config=corr_layers_config,
                    input_rois=input_rois,
                    is_training=is_training)


        elif fc_layers_type == 'fusion_fc_layers':
            corr_layers_config = layers_config.fusion_fc_layers

            corr_out = fusion_corr_layers(
                corr_layers_config=corr_layers_config,
                input_rois=input_rois,
                is_training=is_training)
        else:
            raise ValueError('Invalid fc layers config')

    return corr_out


def basic_corr_layers(corr_layers_config, input_rois, is_training):

    num_layers = corr_layers_config.num_layers
    layer_sizes = corr_layers_config.layer_sizes
    l2_weight_decay = corr_layers_config.l2_weight_decay
    keep_prob = corr_layers_config.keep_prob

    if not num_layers == len(layer_sizes):
        raise ValueError('num_layers does not match length of layer_sizes')

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    with slim.arg_scope([slim.fully_connected], weights_regularizer=weights_regularizer):
        # Flatten
        fc_drop = slim.flatten(input_rois, scope='corr_flatten')
        for layer_idx in range(num_layers):
            fc_name_idx = 6 + layer_idx
            # Use conv2d instead of fully_connected layers.
            fc_layer = slim.fully_connected(fc_drop, layer_sizes[layer_idx],
                                            scope='corr_fc{}'.format(fc_name_idx))

            fc_drop = slim.dropout(fc_layer,
                                   keep_prob=keep_prob,
                                   is_training=is_training,
                                   scope='corr_fc{}_drop'.format(fc_name_idx))

            fc_name_idx += 1

        # [delta_x, delta_z, delta_theta]
        corr_out_size = 3
        corr_out = slim.fully_connected(fc_drop,
                                            corr_out_size,
                                            activation_fn=None,
                                            scope='off_out')
    return corr_out



def fusion_corr_layers(corr_layers_config, input_rois, is_training):
    # Parse configs
    fusion_type = corr_layers_config.fusion_type
    fusion_method = corr_layers_config.fusion_method

    num_layers = corr_layers_config.num_layers
    layer_sizes = corr_layers_config.layer_sizes
    l2_weight_decay = corr_layers_config.l2_weight_decay
    keep_prob = corr_layers_config.keep_prob

    if not len(layer_sizes) == num_layers:
        raise ValueError('Length of layer_sizes does not match num_layers')

    if fusion_type == 'early':
        corr_out = _early_fusion_fc_layers(num_layers=num_layers,
                                    layer_sizes=layer_sizes,
                                    input_rois=input_rois,
                                    l2_weight_decay=l2_weight_decay,
                                    keep_prob=keep_prob,
                                    is_training=is_training)
    # elif fusion_type == 'late':
    #     corr_out = _late_fusion_fc_layers(num_layers=num_layers,
    #                                        layer_sizes=layer_sizes,
    #                                        input_rois=input_rois,
    #                                        input_weights=input_weights,
    #                                        fusion_method=fusion_method,
    #                                        l2_weight_decay=l2_weight_decay,
    #                                        keep_prob=keep_prob,
    #                                        is_training=is_training)
    # elif fusion_type == 'deep':
    #     corr_out = _deep_fusion_fc_layers(num_layers=num_layers,
    #                                        layer_sizes=layer_sizes,
    #                                        input_rois=input_rois,
    #                                        input_weights=input_weights,
    #                                        fusion_method=fusion_method,
    #                                        l2_weight_decay=l2_weight_decay,
    #                                        keep_prob=keep_prob,
    #                                        is_training=is_training)
    else:
        raise ValueError('Invalid fusion type {}'.format(fusion_type))

    return corr_out


def _early_fusion_fc_layers(num_layers, layer_sizes,
                            input_rois, l2_weight_decay,
                            keep_prob, is_training):
    if not num_layers == len(layer_sizes):
        raise ValueError('num_layers does not match length of layer_sizes')

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # Flatten
    fc_drop = slim.flatten(input_rois)

    with slim.arg_scope([slim.fully_connected], weights_regularizer=weights_regularizer):

        for layer_idx in range(num_layers):
            fc_name_idx = 6 + layer_idx

            # Use conv2d instead of fully_connected layers.
            fc_layer = slim.fully_connected(fc_drop, layer_sizes[layer_idx],
                                            scope='fc{}'.format(fc_name_idx))

            # fc_layer = slim.conv2d(fc_drop, layer_sizes[layer_idx],
            #                         [1, 1], scope='fc{}'.format(fc_name_idx))

            fc_drop = slim.dropout(
                fc_layer,
                keep_prob=keep_prob,
                is_training=is_training,
                scope='fc{}_drop'.format(fc_name_idx))

            fc_name_idx += 1

        # correlation out
        # [delta_x, delta_z, delta_theta]
        corr_out_size = 3
        corr_out = slim.fully_connected(fc_drop,
                                        corr_out_size,
                                        activation_fn=None,
                                        scope='off_out')
    return corr_out

def _late_fusion_fc_layers(num_layers, layer_sizes,
                           input_rois, input_weights, fusion_method,
                           l2_weight_decay, keep_prob, is_training):
    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # Build fc layers, one branch per input
    num_branches = len(input_rois)
    branch_outputs = []

    with slim.arg_scope(
            [slim.fully_connected],
            weights_regularizer=weights_regularizer):
        for branch_idx in range(num_branches):

            # Branch feature ROIs
            branch_rois = input_rois[branch_idx]
            fc_drop = slim.flatten(branch_rois,
                                   scope='br{}_flatten'.format(branch_idx))

            for layer_idx in range(num_layers):
                fc_name_idx = 6 + layer_idx

                # Use conv2d instead of fully_connected layers.
                fc_layer = slim.fully_connected(
                    fc_drop, layer_sizes[layer_idx],
                    scope='br{}_fc{}'.format(branch_idx, fc_name_idx))

                fc_drop = slim.dropout(
                    fc_layer,
                    keep_prob=keep_prob,
                    is_training=is_training,
                    scope='br{}_fc{}_drop'.format(branch_idx, fc_name_idx))

            branch_outputs.append(fc_drop)

        # Feature fusion
        fused_features = avod_fc_layer_utils.feature_fusion(fusion_method,
                                                            branch_outputs,
                                                            input_weights)

        # correlation out
        # [delta_x, delta_y, delta_z, delta_theta]
        corr_out_size = 4
        corr_out = slim.fully_connected(fused_features,
                                        corr_out_size,
                                        activation_fn=None,
                                        scope='off_out')
    return corr_out


def _deep_fusion_fc_layers(num_layers, layer_sizes,
                           input_rois, input_weights, fusion_method,
                           l2_weight_decay, keep_prob, is_training):

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # Apply fusion
    fusion_layer = avod_fc_layer_utils.feature_fusion(fusion_method,
                                                      input_rois,
                                                      input_weights)
    fusion_layer = slim.flatten(fusion_layer, scope='flatten')

    with slim.arg_scope(
            [slim.fully_connected],
            weights_regularizer=weights_regularizer):
        # Build layers
        for layer_idx in range(num_layers):
            fc_name_idx = 6 + layer_idx

            all_branches = []
            for branch_idx in range(len(input_rois)):
                fc_layer = slim.fully_connected(
                    fusion_layer, layer_sizes[layer_idx],
                    scope='br{}_fc{}'.format(branch_idx, fc_name_idx))
                fc_drop = slim.dropout(
                    fc_layer,
                    keep_prob=keep_prob,
                    is_training=is_training,
                    scope='br{}_fc{}_drop'.format(branch_idx, fc_name_idx))

                all_branches.append(fc_drop)

            # Apply fusion
            fusion_layer = avod_fc_layer_utils.feature_fusion(fusion_method,
                                                              all_branches,
                                                              input_weights)

        # correlation out
        # [delta_x, delta_y, delta_z, delta_theta]
        corr_out_size = 4
        corr_out = slim.fully_connected(fusion_layer,
                                        corr_out_size,
                                        activation_fn=None,
                                        scope='off_out')
    return corr_out