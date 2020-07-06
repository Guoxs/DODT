from tensorflow.contrib import slim
import tensorflow as tf
from avod.core.corr_layers.correlation import correlation


def build(avod_layers_config, avod_config, bev_rois, is_training):
    fc_layers_config = avod_layers_config.fusion_fc_layers
    num_layers = fc_layers_config.num_layers
    layer_sizes = fc_layers_config.layer_sizes
    l2_weight_decay = fc_layers_config.l2_weight_decay
    keep_prob = fc_layers_config.keep_prob

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # caluate correlation features
    disp = avod_config.avod_proposal_roi_crop_size
    padding = avod_config.avod_proposal_roi_crop_size
    # output_size: [N, 7, 7, 225] // testing [N, 3, 3, 49]
    # roi_corr_feature_maps = correlation(bev_rois[0], bev_rois[1],
    #                                          kernel_size=1,
    #                                          max_displacement=disp,
    #                                          stride_1=1, stride_2=1,
    #                                          padding=padding)

    fusion_feature_maps = tf.concat(bev_rois, axis=1)
    roi_corr_feature_maps = slim.conv2d(fusion_feature_maps, 256, [1, 1], scope='corr_fused')

    output_names = ['corr_cls', 'corr_off']
    cls_logits = None
    offsets = None

    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=weights_regularizer):
        for output in output_names:
            # Flatten
            fc_drop = slim.flatten(roi_corr_feature_maps, scope=output + '_flatten')

            for layer_idx in range(num_layers):
                fc_name_idx = layer_idx
                # Use conv2d instead of fully_connected layers.
                fc_layer = slim.fully_connected(fc_drop, layer_sizes[layer_idx],
                                                scope=output + 'corr_fc{}'.format(fc_name_idx))
                fc_drop = slim.dropout(fc_layer,
                                       keep_prob=keep_prob,
                                       is_training=is_training,
                                       scope=output + 'corr_fc{}_drop'.format(fc_name_idx))
                fc_name_idx += 1

            if output == 'corr_cls':
                # Classification, [0,1],
                corr_out_size = 2
                cls_logits = slim.fully_connected(fc_drop,
                                                  corr_out_size,
                                                  activation_fn=None,
                                                  scope='corr_cls')
            elif output == 'corr_off':
                #[delta_x, delta_z, delta_ry]
                corr_out_size = 3
                offsets = slim.fully_connected(fc_drop,
                                               corr_out_size,
                                               activation_fn=None,
                                               scope='corr_offsets')
    return cls_logits, offsets
