import os
import time
import shutil

import cv2
import numpy as np
import numpy.random as rd
import tensorflow as tf
import tensorflow.layers as tl

from configure import Config
from util import img_util

C = Config('mod_mend_Unet')
T = img_util.Tools()
rd.seed(1943)


def run():
    cv2.namedWindow('beta', cv2.WINDOW_KEEPRATIO)
    img = cv2.imread(os.path.join(C.aerial_dir, 'bellingham1.tif'))
    # cv2.imshow('beta', img)
    # cv2.waitKey(3456)

    channel = 3

    xlen, ylen = img.shape[0:2]
    xmod = xlen % C.size
    ymod = ylen % C.size

    xrnd = int(rd.rand() * xmod)
    yrnd = int(rd.rand() * ymod)

    tf.reset_default_graph()
    inp = tf.placeholder(tf.int8, [None, None, None])
    ten = tf.reshape(inp, (-1, C.size, ylen // C.size, C.size, channel))
    ten = tf.transpose(ten, (0, 2, 1, 3, 4))
    ten = tf.reshape(ten, (-1, C.size, C.size, channel))

    sess = tf.Session()
    for i in range(2 ** 11):
        outs = sess.run(ten, {inp: img[xrnd:xrnd - xmod, yrnd:yrnd - ymod]})
        outs = outs.astype(np.uint8)
        print(i)

    sess.close()

    for out in outs:
        cv2.imshow('beta', out)
        cv2.waitKey(3456)


if __name__ == '__main__':
    run()
