import tensorflow as tf
import numpy as np
import math
from avod.core.correlation import correlation

tf.set_random_seed(0)

BATCH_SIZE = 1
HEIGHT = 10
WIDTH = 10
CHANNELS = 1

NEIGHBORHOOD_SIZE = 5
MAX_DISPLACEMENT = int(math.ceil(NEIGHBORHOOD_SIZE / 2.0))
STRIDE_2 = 2

assert(STRIDE_2 <= NEIGHBORHOOD_SIZE)

# Define two feature maps
fmA = tf.ones((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=tf.float32)
fmB = tf.convert_to_tensor(np.random.randint(10, size=(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)

depth = int(math.floor((2.0 * MAX_DISPLACEMENT + 1) / STRIDE_2) ** 2)

print('Output should be size:', (BATCH_SIZE, HEIGHT, WIDTH, depth))
print('Striding at values: ', [e for e in range(-MAX_DISPLACEMENT + 1, MAX_DISPLACEMENT, STRIDE_2)])

def main():
    sess = tf.Session()
    out = []
    for i in range(-MAX_DISPLACEMENT + 1, MAX_DISPLACEMENT, STRIDE_2): # height
        for j in range(-MAX_DISPLACEMENT + 1, MAX_DISPLACEMENT, STRIDE_2): # width
            padded_a = tf.pad(fmA, [[0,0], [0, abs(i)], [0, abs(j)], [0, 0]])
            padded_b = tf.pad(fmB, [[0, 0], [abs(i), 0], [abs(j), 0], [0, 0]])
            m = padded_a * padded_b
            # if i == MAX_DISPLACEMENT -1 and j == MAX_DISPLACEMENT - 1:
            #     sess = tf.Session()
            #     print(sess.run(padded_a[0,:,:,0]))
            #     print(sess.run(padded_b[0, :, :, 0]))
            #     print(sess.run(m[0,:,:,0]))
            #print(fmA.shape, padded_a.shape, m.shape)

            height_start_idx = 0 if i <= 0 else i
            height_end_idx = height_start_idx + NEIGHBORHOOD_SIZE
            width_start_idx = 0 if j <= 0 else j
            width_end_idx = width_start_idx + NEIGHBORHOOD_SIZE
            cut = m[:, height_start_idx:height_end_idx, width_start_idx:width_end_idx, :]

            final = tf.reduce_sum(cut, 3)
            out.append(final)
    corr = tf.stack(out, 3)
    print('Output size: ', corr.shape)
    print(sess.run(corr[0,:,:,0]))
    out_2 = correlation(fmA, fmA, 1, MAX_DISPLACEMENT, STRIDE_2, STRIDE_2, 3)
    print('Output2 size: ', out_2.shape)
    print(sess.run(out_2[:,:,:,:]))
    # print(sess.run(fmB))
main()