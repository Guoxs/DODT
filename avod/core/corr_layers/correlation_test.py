import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.contrib import slim

from avod.core.corr_layers.correlation import correlation

tf.set_random_seed(0)
np.random.seed(0)

BATCH_SIZE = 1
HEIGHT = 60
WIDTH = 198
CHANNELS = 64


# Define two feature maps
fmA = tf.Variable(np.random.random((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)
fmB = tf.Variable(np.random.random((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)
corr_feature_map = correlation(fmA, fmB, 1, 20, 1, 2, 20)

corr_bottleneck = slim.conv2d(corr_feature_map, 1, [1, 1],
                            scope='bev_corr_bottleneck',
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': True})


def main():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = time.time()
    sess.run(corr_feature_map)
    end = time.time()
    print('correlation time:', end-start)
    n_start = time.time()
    sess.run(corr_bottleneck)
    n_end = time.time()
    print('bottleneck time:', n_end-n_start)
main()