import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.contrib import slim

from avod.core.corr_layers.correlation import correlation

tf.set_random_seed(0)
np.random.seed(0)

# BATCH_SIZE = 1
# HEIGHT = 60
# WIDTH = 198
# CHANNELS = 64
#
#
# # Define two feature maps
# fmA = tf.Variable(np.random.random((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)
# fmB = tf.Variable(np.random.random((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)
#
# corr_feature_map = correlation(fmA, fmB, 1, 20, 1, 2, 20)
#
# corr_bottleneck = slim.conv2d(corr_feature_map, 1, [1, 1],
#                             scope='bev_corr_bottleneck',
#                             normalizer_fn=slim.batch_norm,
#                             normalizer_params={'is_training': True})

concat3_1 = tf.Variable(np.random.random((1, 176, 200, 256)), dtype=tf.float32)
concat3_2 = tf.Variable(np.random.random((1, 176, 200, 256)), dtype=tf.float32)

concat2_1 = tf.Variable(np.random.random((1, 352, 400, 128)), dtype=tf.float32)
concat2_2 = tf.Variable(np.random.random((1, 352, 400, 128)), dtype=tf.float32)

concat1_1 = tf.Variable(np.random.random((1, 704, 800, 64)), dtype=tf.float32)
concat1_2 = tf.Variable(np.random.random((1, 704, 800, 64)), dtype=tf.float32)

corr_3 = correlation(concat3_1, concat3_2, 1, 10, 1, 2, 10)
corr_2 = correlation(concat2_1, concat2_2, 1, 10, 1, 2, 10)
corr_1 = correlation(concat1_1, concat1_2, 1, 10, 1, 2, 10)

upcorr_3 = slim.conv2d_transpose(  corr_3,
                                   128,
                                   [3, 3],
                                   stride=2,
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params={'is_training': True},
                                   scope='upcorr3')

concat_corr_3 = tf.concat((upcorr_3, corr_2), axis=3, name='concat_corr_3')

pyramid_fusion2 = slim.conv2d(concat_corr_3,
                              64,
                              [3, 3],
                              normalizer_fn=slim.batch_norm,
                              normalizer_params={
                                  'is_training': True},
                              scope='pyramid_fusion3')


upcorr_2 = slim.conv2d_transpose(pyramid_fusion2,
                                 64,
                                 [3, 3],
                                  stride=2,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params={'is_training': True},
                                 scope='upcorr2')

concat_corr_2 = tf.concat((upcorr_2, corr_1), axis=3, name='concat_corr_2')


pyramid_corr_fusion = slim.conv2d(concat_corr_2,
                                  32,
                                  [3,3],
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params={
                                      'is_training': True},
                                  scope='pyramid_fusion1')

pyramid_corr_fusion = pyramid_corr_fusion[:, 4:]


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.shape(pyramid_fusion2)))
print(sess.run(tf.shape(pyramid_corr_fusion)))
print(sess.run(tf.shape(concat_corr_3)))
