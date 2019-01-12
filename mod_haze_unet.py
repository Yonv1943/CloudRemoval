import os
import time
import shutil

import numpy as np
import tensorflow as tf
import tensorflow.layers as tl

from utils import mod_util
from utils import img_util

"""
2018-10-10 Yonv1943
    Reference: https://github.com/jiamings/wgan
    Reference: https://github.com/cameronfabbri/Improved-Wasserstein-GAN
    Reference: https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN
2018-04-03 Yonv1943
2018-09-27 tf.layers.conv2d
2018-11-28 tf.data, stable
2018-12-19 u-net, reconstruction
2018-12-22 loss
2018-12-28 size == 2 ** 8, train_size == 2 ** 19
"""


class Config(object):
    train_epoch = 2 ** 11
    train_size = int(2 ** 17 * 0.99)
    eval_size = 2 ** 4 - 2  # 2 ** 3
    batch_size = int(2 ** 6)
    batch_epoch = train_size // batch_size

    size = int(2 ** 8)
    replace_num = int(0.368 * batch_size)
    learning_rate = 8e-5  # 1e-4

    show_gap = 2 ** 7  # time
    eval_gap = 2 ** 10  # time
    gpu_limit = 0.8  # 0.0 ~ 1.0
    gpu_id = 0

    data_dir = '/mnt/sdb1/data_sets'
    aerial_dir = os.path.join(data_dir, 'AerialImageDataset/train')
    cloud_dir = os.path.join(data_dir, 'ftp.nnvl.noaa.gov_color_IR_2018')
    grey_dir = os.path.join(data_dir, 'CloudGreyDataset')

    def __init__(self, model_dir='mod'):
        self.model_dir = model_dir
        self.model_name = 'mod'
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.model_npz = os.path.join(self.model_dir, self.model_name + '.npz')
        self.model_log = os.path.join(self.model_dir, 'training_npy.txt')


if __name__ != '__main__':
    from configure import Config  # for test

    print("|| TEST")

C = Config('mod_haze_unet')
tf.set_random_seed(time.time() * 1943 % 178320049)


def unet(inp0, dim, out_dim, name, reuse):
    def leru_batch_norm(ten):
        ten = tf.layers.batch_normalization(ten, training=True)
        ten = tf.nn.leaky_relu(ten)
        return ten

    paddings = tf.constant(((0, 0), (1, 1), (1, 1), (0, 0)))

    def conv_pad(ten, idx, step=2):
        filters = (2 ** idx) * dim
        ten = tf.pad(ten, paddings, 'REFLECT')
        ten = tl.conv2d(ten, filters, 3, step, 'valid', activation=tf.nn.leaky_relu)
        return ten

    def conv_tp(ten, idx):
        filters = (2 ** idx) * dim
        ten = tl.conv2d_transpose(ten, filters * 2, 3, 2, 'same', activation=leru_batch_norm)
        return ten

    def conv_tp_conv(ten, idx, step0=1, step1=1):
        filters = (2 ** idx) * dim
        ten = tl.conv2d_transpose(ten, filters, 3, step0, 'valid', activation=leru_batch_norm)
        ten = tl.conv2d(ten, filters, 3, step1, 'valid', activation=tf.nn.relu6)
        return ten

    with tf.variable_scope(name, reuse=reuse):
        ten1 = conv_pad(inp0, 0)
        ten2 = conv_pad(ten1, 1)
        ten3 = conv_pad(ten2, 2)
        ten4 = conv_pad(ten3, 3)
        ten5 = conv_pad(ten4, 4)

        ten6 = conv_tp_conv(ten5, 4, 1, 1) + ten5
        ten5 = conv_tp_conv(ten6, 4, 1, 1) + ten6

        ten4 = tf.concat((conv_tp(ten5, 4), ten4), axis=3)
        ten3 = tf.concat((conv_tp(ten4, 4), ten3), axis=3)
        ten2 = tf.concat((conv_tp(ten3, 3), ten2), axis=3)
        ten1 = tf.concat((conv_tp(ten2, 2), ten1), axis=3)
        ten0 = tf.concat((conv_tp(ten1, 1), inp0), axis=3)

        ten0 = conv_pad(ten0, idx=0, step=1)
        ten0 = tl.conv2d(ten0, out_dim, 1, 1, 'same', activation=tf.nn.sigmoid)
        return ten0


def init_train():
    # tf.reset_default_graph()
    unet_name, unet_dim = 'unet', 16

    '''init'''
    inp_ground = tf.placeholder(tf.uint8, [None, C.size, C.size, 3])
    ten_ground = tf.to_float(inp_ground)
    ten_ground *= tf.random_uniform([], 0.00382, 0.00402) + tf.random_uniform([1, 1, 1, 3], -0.00012, 0.00012)
    ten_ground += tf.random_uniform([], -0.02, 0.02) + tf.random_uniform([1, 1, 1, 3], -0.02, 0.02)
    ten_ground = tf.clip_by_value(ten_ground, 0, 1)

    inp_mask01 = tf.placeholder(tf.uint8, [None, C.size, C.size, 1])
    ten_mask01 = tf.to_float(inp_mask01)
    ten_mask01 *= tf.random_uniform([], 0.00382, 0.00402)
    ten_mask01 += tf.random_uniform([], -0.02, 0.02)
    ten_mask01 = tf.clip_by_value(ten_mask01, 0, 1)

    '''func'''
    ten_mask10 = 1.0 - ten_mask01
    ten_aerial = ten_ground * ten_mask10 + ten_mask01

    ten_grdcld = unet(ten_aerial, unet_dim, 4, unet_name, reuse=False)
    out_ground = ten_grdcld[:, :, :, 0:3]
    out_mask01 = ten_grdcld[:, :, :, 3:4]

    '''loss'''
    loss_mask01 = (ten_mask01 - out_mask01) ** 2
    loss_mask01 = tf.reduce_mean(loss_mask01)

    loss_aerial = (ten_ground - out_ground) ** 2
    loss_aerial *= tf.clip_by_value(ten_mask10 * 2 - 0.5, 0, 1)
    loss_aerial = tf.reduce_mean(loss_aerial)

    loss_haze = loss_mask01 + loss_aerial

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        tf_vars = tf.trainable_variables()

        optz_haze = tf.train.AdamOptimizer(learning_rate=C.learning_rate, beta1=0.5, beta2=0.9)
        optz_haze = optz_haze.minimize(loss_haze, var_list=[v for v in tf_vars if v.name.startswith(unet_name)])

        loss = [loss_mask01, loss_aerial]
        optz = [optz_haze, loss_aerial]

    train_fetch = [loss, optz]
    eval_fetch = [ten_aerial, ten_ground, out_ground, ten_mask01, out_mask01]
    return inp_ground, inp_mask01, train_fetch, eval_fetch


def process_train(feed_queue):
    print("||Training Initialize")
    inp_ground, inp_mask01, train_fetch, eval_fetch = init_train()

    sess = mod_util.get_sess(C)
    saver, logger, pre_epoch = mod_util.get_saver_logger(C, sess)
    print("||Training Check")
    eval_list = feed_queue.get()
    eval_feed_dict = {inp_ground: eval_list[0],
                      inp_mask01: eval_list[1], }
    sess.run(eval_fetch, eval_feed_dict)

    print("||Training Start")
    start_time = show_time = eval_time = time.time()
    loss = (0, 0)
    try:
        for epoch in range(C.train_epoch):
            for i in range(C.batch_size):
                batch_data = feed_queue.get()
                batch_dict = {inp_ground: batch_data[0],
                              inp_mask01: batch_data[1], }
                loss, optz = sess.run(train_fetch, batch_dict)

            logger.write('%e %e\n' % (loss[0], loss[1]))

            if time.time() - show_time > C.show_gap:
                show_time = time.time()
                remain_epoch = C.train_epoch - epoch
                remain_time = (show_time - start_time) * remain_epoch / (epoch + 1)
                print(end="\n|  %3d s |%3d epoch | Loss: %9.3e %9.3e"
                          % (remain_time, remain_epoch, loss[0], loss[1]))
            if time.time() - eval_time > C.eval_gap:
                eval_time = time.time()
                logger.close()
                logger = open(C.model_log, 'a')

                eval_feed_dict[inp_mask01] = np.rot90(eval_feed_dict[inp_mask01], axes=(1, 2))
                img_util.get_eval_img(mat_list=sess.run(eval_fetch, eval_feed_dict), channel=3,
                                      img_path="%s/eval-%08d.jpg" % (C.model_dir, pre_epoch + epoch))
                print(end="  EVAL %d" % (pre_epoch + epoch))

    except KeyboardInterrupt:
        print("\n| KeyboardInterrupt:", process_train.__name__)
    print("\n")
    print('  Train_epoch: %d' % C.train_epoch)
    print('  Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    print('  TimeUsed:    %d' % int(time.time() - start_time))

    img_util.get_eval_img(mat_list=sess.run(eval_fetch, eval_feed_dict), channel=3,
                          img_path="%s/eval-%08d.jpg" % (C.model_dir, 0))
    saver.save(sess, C.model_path, write_meta_graph=False)
    print("  SAVE: %s" % C.model_path)
    mod_util.save_npy(sess, C.model_npz)

    logger.close()
    sess.close()

    mod_util.draw_plot(C.model_log)


def process_feed(feed_queue):
    ts = C.train_size
    bs = C.batch_size
    rd = np.random
    rd_randint = rd.randint
    rd_shuffle = rd.shuffle

    timer = time.time()
    grounds = img_util.get_data__ground(ts, channel=3)
    print("||Data_sets: ready for check. Used time:", int(time.time() - timer))
    mask01s = img_util.get_data__cloud1(ts)
    print("||Data_sets: ready for check. Used time:", int(time.time() - timer))

    eval_id = np.random.randint(ts // 2, ts, C.eval_size * 4)
    eval_id = list(set(eval_id))[:C.eval_size]
    feed_queue.put([grounds[eval_id],
                    mask01s[eval_id], ])  # for eval

    print("||Data_sets: ready for training")
    i0_range = np.arange(C.batch_epoch)
    i1_range = np.arange(C.batch_epoch)
    try:
        for epoch in range(C.train_epoch):
            rd_shuffle(i0_range)
            rd_shuffle(i1_range)
            for i0, i1 in zip(i0_range, i1_range):
                j = i0 * bs
                k = i1 * bs

                switch = rd_randint(4)
                if switch == 0:
                    grounds[j:j + bs] = np.rot90(grounds[j:j + bs], axes=(1, 2))
                elif switch == 1:
                    grounds[j:j + bs] = grounds[j:j + bs, ::-1]  # image flipped
                elif switch == 2:
                    mask01s[j:j + bs] = np.rot90(mask01s[j:j + bs], axes=(1, 2))
                elif switch == 3:
                    mask01s[j:j + bs] = mask01s[j:j + bs, ::-1]  # image flipped

                feed_queue.put([grounds[j:j + bs],
                                mask01s[k:k + bs], ])

    except KeyboardInterrupt:
        print("| KeyboardInterrupt:", process_feed.__name__)
    print("| quit:", process_feed.__name__)


def run():
    print('| Train_epoch: %d' % C.train_epoch)
    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    if input("||PRESS 'y' to REMOVE model_dir? %s: " % C.model_dir) == 'y':
        shutil.rmtree(C.model_dir, ignore_errors=True)
        print("||Remove")
    elif input("||PRESS 'y' to UPDATE model_npz? %s: " % C.model_npz) == 'y':
        mod_util.update_npz(src_path='mod_AutoEncoder/mod.npz', dst_path=C.model_npz)

        remove_path = os.path.join(C.model_dir, 'checkpoint')
        os.remove(remove_path) if os.path.exists(remove_path) else None

    import multiprocessing as mp
    feed_queue = mp.Queue(maxsize=8)
    process = [mp.Process(target=process_feed, args=(feed_queue,)),
               mp.Process(target=process_train, args=(feed_queue,)), ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run()
