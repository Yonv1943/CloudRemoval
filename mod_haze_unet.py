import os
import time
import shutil

import numpy as np
import tensorflow as tf
import tensorflow.layers as tl

from utils import mod_util
from utils import img_util

"""
2018-04-03 Yonv1943
2018-09-27 tf.layers.conv2d
2018-11-28 tf.data, stable
Source: https://github.com/aymericdamien/TensorFlow-Examples/
"""


class Config(object):
    train_epoch = 2 ** 14
    train_size = int(2 ** 17 * 1.9)
    eval_size = 2 ** 3
    batch_size = 2 ** 4
    batch_epoch = train_size // batch_size

    size = int(2 ** 7)
    replace_num = int(0.25 * batch_size)

    show_gap = 2 ** 4  # time
    eval_gap = 2 ** 7  # time
    gpu_limit = 0.48  # 0.0 ~ 1.0
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

# from configure import Config
C = Config('mod_haze_unet')
tf.set_random_seed(1943)


def leru_batch_norm(ten):
    ten = tf.layers.batch_normalization(ten, training=True)
    ten = tf.nn.leaky_relu(ten)
    return ten


# def haze_unet(inp0, dim, name, reuse):
#     def conv(ten, idx):
#         filters = (2 ** idx) * dim
#         return tl.conv2d(ten, filters, 3, 2, 'same', activation=tf.nn.leaky_relu)
#
#     def conv_tp_conv(ten, idx):
#         filters = (2 ** idx) * dim
#         ten = tl.conv2d_transpose(ten, filters, 3, 1, 'valid', activation=leru_batch_norm)
#         ten = tl.conv2d(ten, filters, 3, 1, 'valid', activation=tf.nn.leaky_relu)
#         return ten
#
#     def conv_tp_concat(ten, con, idx):
#         filters = (2 ** idx) * dim
#         ten = tl.conv2d_transpose(ten, filters, 3, 2, 'same', activation=leru_batch_norm)
#         return tf.concat((ten, con), axis=3)
#
#     with tf.variable_scope(name, reuse=reuse):
#         ten1 = conv(inp0, 0)
#         ten2 = conv(ten1, 1)
#         ten3 = conv(ten2, 2)
#         ten4 = conv(ten3, 3)
#         ten5 = conv(ten4, 4)
#
#         ten6 = conv_tp_conv(ten5, 4) + ten5
#         ten7 = conv_tp_conv(ten6, 4) + ten6
#
#         ten4 = conv_tp_concat(ten7, ten4, 5)
#         ten3 = conv_tp_concat(ten4, ten3, 4)
#         ten2 = conv_tp_concat(ten3, ten2, 3)
#         ten1 = conv_tp_concat(ten2, ten1, 2)
#         ten0 = conv_tp_concat(ten1, inp0, 1)
#
#         ten0 = conv_tp_conv(ten0, 0)
#         ten0 = tl.conv2d(ten0, 4, 1, 1, 'same', activation=tf.nn.tanh)
#         return ten0 * 0.505 + 0.5
def haze_unet(inp0, dim, name, reuse):
    def conv_tp_conv(ten, idx, step0=1, step1=2):
        filters = (2 ** idx) * dim
        ten = tl.conv2d_transpose(ten, filters, 3, step0, 'valid', activation=leru_batch_norm)
        ten = tl.conv2d(ten, filters, 3, step1, 'valid', activation=tf.nn.leaky_relu)
        return ten

    def conv_tp_conv_concat(ten, con, idx, step0=2, step1=1):
        filters = (2 ** idx) * dim
        ten = tl.conv2d_transpose(ten, filters, 3, step0, 'valid', activation=leru_batch_norm)
        ten = tl.conv2d(ten, filters, 3, step1, 'same', activation=tf.nn.leaky_relu)
        return tf.concat((ten[:, :-1, :-1], con), axis=3)

    with tf.variable_scope(name, reuse=reuse):
        ten1 = conv_tp_conv(inp0, 0, 1, 2)
        ten2 = conv_tp_conv(ten1, 1, 1, 2)
        ten3 = conv_tp_conv(ten2, 2, 1, 2)
        ten4 = conv_tp_conv(ten3, 3, 1, 2)
        ten5 = conv_tp_conv(ten4, 4, 1, 2)

        ten6 = conv_tp_conv(ten5, 4, 1, 1) + ten5
        ten7 = conv_tp_conv(ten6, 4, 1, 1) + ten6

        ten4 = conv_tp_conv_concat(ten7, ten4, 4)
        ten3 = conv_tp_conv_concat(ten4, ten3, 3)
        ten2 = conv_tp_conv_concat(ten3, ten2, 2)
        ten1 = conv_tp_conv_concat(ten2, ten1, 1)
        ten0 = conv_tp_conv_concat(ten1, inp0, 0)

        ten0 = conv_tp_conv(ten0, 0, 1, 1)
        ten0 = tl.conv2d(ten0, 4, 1, 1, 'same', activation=tf.nn.tanh)
        return ten0 * 0.505 + 0.5


def process_train(feed_queue):
    # tf.reset_default_graph()
    name_unet = 'unet'

    print("||Training Initialize")
    inp_ground = tf.placeholder(tf.uint8, [None, C.size, C.size, 3])
    inp_cloud1 = tf.placeholder(tf.uint8, [None, C.size, C.size, 1])
    ten_repeat = tf.ones([1, 1, 1, 3])

    ten_ground = tf.to_float(inp_ground) / 255.0
    ten_cloud1 = tf.clip_by_value(tf.random_uniform([], 0.00365, 0.00415) * tf.to_float(inp_cloud1) +
                                  tf.random_uniform([], -0.1, +0.1), 0.0, 1.0)
    # ten_cloud1 = tf.to_float(inp_cloud1) / 255.0
    # ten_cloud1 = tf.clip_by_value(tf.random_uniform([], 0.9, 1.1) * ten_cloud1 +
    #                               tf.random_uniform([], -0.1, +0.1),
    #                               0.0, 1.0)
    ten_cloud3 = ten_cloud1 * ten_repeat
    ten_mask10 = (1.0 - ten_cloud3)
    ten_aerial = ten_ground * ten_mask10 + ten_cloud3

    ten_grdcld = haze_unet(ten_aerial, 24, name_unet, reuse=False)
    out_ground = ten_grdcld[:, :, :, 0:3]
    out_cloud1 = ten_grdcld[:, :, :, 3:4]
    out_cloud3 = out_cloud1 * ten_repeat

    loss_cloud1 = tf.losses.mean_pairwise_squared_error(ten_cloud1, out_cloud1)
    loss_aerial = tf.losses.mean_pairwise_squared_error(ten_ground * ten_mask10,
                                                        out_ground * ten_mask10)
    loss_haze = loss_cloud1 + loss_aerial

    tf_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optz_haze = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_haze, var_list=[v for v in tf_vars if v.name.startswith(name_unet)])
    loss = [loss_cloud1, loss_aerial]
    optz = [optz_haze, ]

    sess = mod_util.get_sess(C)
    saver, logger, pre_epoch = mod_util.get_saver_logger(C, sess)
    print("||Training Check")
    eval_list = feed_queue.get()
    eval_fetch = [ten_aerial, ten_ground, out_ground, ten_cloud3, out_cloud3]
    eval_feed_dict = {inp_ground: eval_list[0],
                      inp_cloud1: eval_list[1]}
    sess.run([loss, optz], eval_feed_dict)

    print("||Training Start")
    start_time = show_time = eval_time = time.time()
    try:
        for epoch in range(C.train_epoch):
            batch_losses = list()  # init
            for i in range(C.batch_size):
                feed_data = feed_queue.get()
                batch_return = sess.run([loss, optz], {inp_ground: feed_data[0],
                                                       inp_cloud1: feed_data[1]})
                batch_losses.append(batch_return[0])

            loss_average = np.mean(batch_losses, axis=0)
            loss_error = np.std(batch_losses, axis=0)
            logger.write('%e %e %e %e\n' % (loss_average[0], loss_error[0],
                                            loss_average[1], loss_error[1],))

            if time.time() - show_time > C.show_gap:
                show_time = time.time()
                remain_epoch = C.train_epoch - epoch
                remain_time = (show_time - start_time) * remain_epoch / (epoch + 1)
                print(end="\n|  %3d s |%3d epoch | Loss: %9.3e %9.3e"
                          % (remain_time, remain_epoch, loss_average[0], loss_average[1]))
            elif time.time() - eval_time > C.eval_gap:
                eval_time = time.time()
                logger.close()
                logger = open(C.model_log, 'a')

                img_util.get_eval_img(mat_list=sess.run(eval_fetch, eval_feed_dict), channel=3,
                                      img_path="%s/eval-%08d.jpg" % (C.model_dir, pre_epoch + epoch))
                print(end="  EVAL")

    except KeyboardInterrupt:
        print("\n| KeyboardInterrupt:", process_train.__name__)
    print("\n")
    print('| Train_epoch: %d' % C.train_epoch)
    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    print('| TimeUsed:    %d' % int(time.time() - start_time))

    img_util.get_eval_img(mat_list=sess.run(eval_fetch, eval_feed_dict), channel=3,
                          img_path="%s/eval-%08d.jpg" % (C.model_dir, 0))
    saver.save(sess, C.model_path, write_meta_graph=False)
    mod_util.save_npy(sess, C.model_npz)
    print("| SAVE: %s" % C.model_path)

    logger.close()
    sess.close()

    mod_util.draw_plot(C.model_log)


def process_data(feed_queue):
    ts = C.train_size
    bs = C.batch_size

    timer = time.time()
    grounds = img_util.get_data__ground(ts, channel=3)
    cloud1s = img_util.get_data__cloud1(ts)
    print("  load data: %d sec" % (time.time() - timer))

    print("||Data_sets: ready for check |Used time:", int(time.time() - timer))
    eval_id = np.random.randint(ts // 2, ts, C.eval_size * 2)
    eval_id = list(set(eval_id))[:C.eval_size]
    feed_queue.put([grounds[eval_id],
                    cloud1s[eval_id]])  # for eval

    print("||Data_sets: ready for training")
    i0_range = np.arange(C.batch_epoch)
    i1_range = np.arange(C.batch_epoch)
    try:
        for epoch in range(C.train_epoch):
            np.random.shuffle(i0_range)
            np.random.shuffle(i1_range)
            for i0, i1 in zip(i0_range, i1_range):
                j = i0 * bs
                feed_queue.put([grounds[j:j + bs],
                                cloud1s[j:j + bs], ])
    except KeyboardInterrupt:
        print("| KeyboardInterrupt:", process_data.__name__)
    print("| quit:", process_data.__name__)


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
    process = [mp.Process(target=process_data, args=(feed_queue,)),
               mp.Process(target=process_train, args=(feed_queue,)), ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run()
