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
"""


class Config(object):
    train_epoch = 2 ** 11
    train_size = int(2 ** 17 * 1.2)
    eval_size = 2 ** 3
    batch_size = int(2 ** 5)
    batch_epoch = train_size // batch_size

    size = int(2 ** 8)
    replace_rate = int(0.5 * batch_size)
    learning_rate = 8e-5  # 1e-4

    show_gap = 2 ** 7  # time
    eval_gap = 2 ** 10  # time
    gpu_limit = 0.8  # 0.0 ~ 1.0
    gpu_id = 1

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

    print("||TEST")

C = Config('mod_cloud_detect')
tf.set_random_seed(time.time() * 1943 % 178320049)


def unet(inp0, dim, out_dim, name, reuse, training=True):
    def leru_batch_norm(ten):
        ten = tf.layers.batch_normalization(ten, training=training)
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

    def conv_res(ten, idx):
        return conv_pad(conv_pad(ten, idx, 1), idx, 1) + ten

    with tf.variable_scope(name, reuse=reuse):
        ten1 = conv_pad(inp0, 0)
        ten2 = conv_pad(ten1, 1)
        ten3 = conv_pad(ten2, 2)
        ten4 = conv_pad(ten3, 3)
        ten5 = conv_pad(ten4, 4)

        ten5 = conv_res(ten5, 4)

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
    unet_name, unet_dim = 'unet', 24

    '''init'''
    inp_ground = tf.placeholder(tf.uint8, [None, C.size, C.size, 3])
    ten_ground = tf.to_float(inp_ground)
    ten_ground *= tf.random_uniform([], 0.00382, 0.00402) + tf.random_uniform([1, 1, 1, 3], -0.00012, 0.00012)
    ten_ground += tf.random_uniform([], -0.01, 0.01) + tf.random_uniform([1, 1, 1, 3], -0.01, 0.01)
    ten_ground = tf.clip_by_value(ten_ground, 0, 1)

    inp_mask01 = tf.placeholder(tf.uint8, [None, C.size, C.size, 1])
    ten_mask01 = tf.to_float(inp_mask01)
    ten_mask01 *= tf.random_uniform([], 0.00382, 0.00402)
    ten_mask01 += tf.random_uniform([], -0.05, 0.25)
    ten_mask01 = tf.clip_by_value(ten_mask01, 0, 1)

    '''func'''
    ten_mask10 = 1.0 - ten_mask01
    ten_mask03 = ten_mask01 * tf.random_uniform((1, 1, 1, 3), 0.98, 1.00)
    ten_aerial = ten_ground * ten_mask10 + ten_mask03

    ten_grdcld = unet(ten_aerial, unet_dim, 4, unet_name, reuse=False)
    out_ground = ten_grdcld[:, :, :, 0:3]
    out_mask01 = ten_grdcld[:, :, :, 3:4]

    '''loss'''
    loss_mask01 = tf.reduce_mean((ten_mask01 - out_mask01) ** 2)

    loss_ground = (ten_ground - out_ground) ** 2
    loss_ground *= tf.clip_by_value(ten_mask10 * 1.25 - 0.25, 0, 1)
    loss_ground = tf.reduce_mean(loss_ground)

    loss_aerial = loss_mask01 + loss_ground

    '''optz'''
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        tf_vars = tf.trainable_variables()

        optz_aerial = tf.train.AdamOptimizer(learning_rate=C.learning_rate, beta1=0.5, beta2=0.9) \
            .minimize(loss_aerial, var_list=[v for v in tf_vars if v.name.startswith(unet_name)])

    train_fetch = [[loss_mask01, loss_ground], optz_aerial]

    eva_grdcld = unet(ten_aerial, unet_dim, 4, unet_name, reuse=True, training=False)
    eva_ground = eva_grdcld[:, :, :, 0:3]
    eva_mask01 = eva_grdcld[:, :, :, 3:4]
    eval_fetch = [ten_aerial, eva_ground, eva_mask01]
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
                print(end="\n%3d s |%3d epoch | Loss: %9.3e %9.3e"
                          % (remain_time, remain_epoch, loss[0], loss[1]))

            if time.time() - eval_time > C.eval_gap:
                eval_time = time.time()
                logger.close()  # write info the disk
                logger = open(C.model_log, 'a')

                eval_feed_dict[inp_mask01] = np.rot90(eval_feed_dict[inp_mask01], axes=(1, 2))
                img_util.get_eval_img(mat_list=sess.run(eval_fetch, eval_feed_dict), channel=3,
                                      img_path="%s/eval-%08d.jpg" % (C.model_dir, pre_epoch + epoch))
                print(end="  EVAL %d" % (pre_epoch + epoch))

            if os.path.exists(os.path.join(C.model_dir, 'SAVE.MARK')):
                os.remove(os.path.join(C.model_dir, 'SAVE.MARK'))
                print("\n||Break Training and save:", process_train.__name__)
                break

    except KeyboardInterrupt:
        print("\n||Break Training and save:", process_train.__name__)
    print('\n  TimeUsed:    %d' % int(time.time() - start_time))
    saver.save(sess, C.model_path, write_meta_graph=False)
    print("  SAVE: %s" % C.model_path)
    img_util.get_eval_img(mat_list=sess.run(eval_fetch, eval_feed_dict), channel=3,
                          img_path="%s/eval-%08d.jpg" % (C.model_dir, 0))

    logger.close()
    sess.close()

    os.rmdir(os.path.join(C.model_dir, 'TRAINING.MARK'))


def process_feed(feed_queue):
    ts = C.train_size
    bs = C.batch_size

    rd_randint = np.random.randint
    rd_shuffle = np.random.shuffle

    timer = time.time()

    grounds = img_util.get_data__ground(ts)
    print("  Dataset grounds. Used time:", int(time.time() - timer))
    mask01s = img_util.get_data__cloud1(ts)
    print("  Dataset mask01s. Used time:", int(time.time() - timer))

    eval_id = list(set(np.random.randint(0, ts, C.eval_size * 4)))[:C.eval_size]
    feed_queue.put([grounds[eval_id],
                    mask01s[eval_id], ])  # for eval

    print("  Dataset Ready.   Used time:", int(time.time() - timer))
    i0_range = np.arange(C.batch_epoch)
    i1_range = np.arange(C.batch_epoch)
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
                grounds[j:j + bs] = np.flip(grounds[j:j + bs], axis=rd_randint(1, 3))
            elif switch == 2:
                mask01s[j:j + bs] = np.rot90(mask01s[j:j + bs], axes=(1, 2))
            elif switch == 3:
                mask01s[j:j + bs] = np.flip(mask01s[j:j + bs], axis=rd_randint(1, 3))

            feed_queue.put([grounds[j:j + bs],
                            mask01s[k:k + bs], ])


def run():
    print('||GPUid: %d' % C.gpu_id)
    print('||Epoch: %d' % C.train_epoch)
    print('||Batch: %d' % C.batch_size)
    print('||Model: %s' % C.model_dir)
    if input("||PRESS: 'y' to REMOVE? ") == 'y':
        shutil.rmtree(C.model_dir, ignore_errors=True)
        print("||Remove")
    # elif input("||PRESS 'y' to UPDATE model_npz? %s: " % C.model_npz) == 'y':
    #     # mod_util.save_npy(sess, C.model_npz)
    #     # mod_util.draw_plot(C.model_log)
    #
    #     mod_util.update_npz(src_path='mod_AutoEncoder/mod.npz', dst_path=C.model_npz)
    #
    #     remove_path = os.path.join(C.model_dir, 'checkpoint')
    #     os.remove(remove_path) if os.path.exists(remove_path) else None

    import multiprocessing as mp
    feed_queue = mp.Queue(maxsize=8)
    process = [mp.Process(target=process_feed, args=(feed_queue,)),
               mp.Process(target=process_train, args=(feed_queue,)), ]

    os.makedirs(os.path.join(C.model_dir, 'TRAINING.MARK'), exist_ok=True)
    [p.start() for p in process]

    # [p.join() for p in process]
    while os.path.exists(os.path.join(C.model_dir, 'TRAINING.MARK')):
        time.sleep(2)
    else:
        [p.terminate() for p in process]


if __name__ == '__main__':
    run()
