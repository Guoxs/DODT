import tensorflow as tf
import numpy as np
import math
import time
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

def main():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out_2 = correlation(fmA, fmB, 1, 20, 1, 2, 20)
    print('Output2 size: ', out_2.shape)
    print(sess.run(out_2[:,:,:,:]))
    # print(sess.run(fmB))
main()