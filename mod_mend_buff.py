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
2018-10-11 save eval jpg
2018-10-12 'TF_CPP_MIN_LOG_LEVEL' tf.Session()
2018-10-12 origin, tensorflow.contrib.layers --> tf.layers
2018-10-22 change mask from 'middle square' to 'spot'
2018-10-23 spot --> polygon
2018-10-23 for discriminator, tf.concat([tenx, mask], axis=0)
2018-11-25 kernel3 better than kernel2, but little grid
2018-11-29 load uint8
2018-12-07 resize, buff
2018-12-15 beta, data_feed,
2018-12-15 simplify, reconstruction
2018-12-18 auto-encoder update
"""


class Config(object):
    train_epoch = 2 ** 13
    train_size = int(2 ** 17 * 1.9)
    eval_size = 2 ** 4 - 2  # 2 ** 3
    batch_size = 2 ** 4
    batch_epoch = train_size // batch_size

    size = int(2 ** 7)
    replace_num = int(0.368 * batch_size)
    learning_rate = 8e-5  # 1e-4

    show_gap = 2 ** 7  # time
    eval_gap = 2 ** 10  # time
    gpu_limit = 0.8  # 0.0 ~ 1.0
    gpu_id = 1

    data_dir = '/mnt/sdb1/data_sets'
    aerial_dir = os.path.join(data_dir, 'AerialImageDataset/test')
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

C = Config('mod_mend_buff')
tf.set_random_seed(1943)


def auto_encoder(inp0, dim, out_dim, name, reuse):
    def leru_batch_norm(ten):
        ten = tf.layers.batch_normalization(ten, training=True)
        ten = tf.nn.leaky_relu(ten)
        return ten

    paddings = tf.constant(((0, 0), (1, 1), (1, 1), (0, 0)))
    def conv_pad(ten, idx):
        filters = (2 ** idx) * dim
        ten = tf.pad(ten, paddings, 'REFLECT')
        ten = tl.conv2d(ten, filters, 3, 2, 'valid', activation=tf.nn.leaky_relu)
        return ten

    def conv_tp(ten, idx):
        filters = (2 ** idx) * dim
        ten = tl.conv2d_transpose(ten, filters, 3, 2, 'same', activation=leru_batch_norm)
        return ten

    def conv_tp_conv(ten, idx, step0=1, step1=2):
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
        ten6 = conv_tp_conv(ten5, 4, 1, 1) + ten5
        ten5 = conv_tp_conv(ten6, 4, 1, 1) + ten6
        ten6 = conv_tp_conv(ten5, 4, 1, 1) + ten5
        ten5 = conv_tp_conv(ten6, 4, 1, 1) + ten6
        ten6 = conv_tp_conv(ten5, 4, 1, 1) + ten5
        ten5 = conv_tp_conv(ten6, 4, 1, 1) + ten6

        ten4 = conv_tp(ten5, 4)
        ten3 = conv_tp(ten4, 3)
        ten2 = conv_tp(ten3, 2)
        ten1 = conv_tp(ten2, 1)
        ten0 = conv_tp(ten1, 0)

        ten0 = conv_tp_conv(ten0, 0, 1, 1)
        ten0 = tf.concat((ten0, inp0), axis=3)
        ten0 = tl.conv2d(ten0, out_dim, 1, 1, 'same', activation=tf.nn.sigmoid)
        return ten0


def init_train():
    # tf.reset_default_graph()
    gene_name = 'gene'
    disc_name = 'disc'
    resize = (C.size // 2, C.size // 2)
    resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

    inp_ground = tf.placeholder(tf.uint8, [None, C.size, C.size, 3])
    inp_mask01 = tf.placeholder(tf.uint8, [None, C.size, C.size, 1])
    inp_grdbuf = tf.placeholder(tf.uint8, [None, C.size, C.size, 3])
    inp_mskbuf = tf.placeholder(tf.uint8, [None, C.size, C.size, 1])

    ten_ground = tf.to_float(inp_ground) / 255.0
    ten_mask01 = tf.to_float(inp_mask01) / 255.0
    ten_grdbuf = tf.to_float(inp_grdbuf) / 255.0
    ten_mskbuf = tf.to_float(inp_mskbuf) / 255.0

    ten_repeat = tf.ones([1, 1, 1, 3])
    ten_mask03 = ten_mask01 * ten_repeat
    ten_mask30 = (1.0 - ten_mask03)
    ten_ragged = ten_ground * ten_mask30

    ten_ragge4 = tf.concat((ten_ragged - ten_mask03, ten_mskbuf), axis=3)
    ten_patch3 = auto_encoder(ten_ragge4, 32, 3, gene_name, reuse=False)
    out_ground = ten_ragged + ten_patch3 * ten_mask03

    # real, fake, buff
    ten_refabu = tf.concat((out_ground, ten_grdbuf, ten_ground), axis=0)
    dis_refabu = auto_encoder(ten_refabu, 32, 1, disc_name, reuse=False)

    dis_fake_1 = dis_refabu[:C.batch_size]
    zero_label = tf.zeros_like(dis_fake_1)

    loss_gene = tf.losses.mean_pairwise_squared_error(dis_fake_1, zero_label) * 2 + \
                tf.losses.mean_pairwise_squared_error(
                    ten_patch3, ten_ground,
                    # tf.image.resize_images(ten_patch3, resize, resize_method),
                    # tf.image.resize_images(ten_ground, resize, resize_method),
                )

    loss_disc = tf.losses.mean_pairwise_squared_error(
        dis_refabu, tf.concat((ten_mask01, ten_mskbuf, zero_label), axis=0))

    tf_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optz_gene = tf.train.AdamOptimizer(C.learning_rate, beta1=0.5, beta2=0.9) \
            .minimize(loss_gene, var_list=[v for v in tf_vars if v.name.startswith(gene_name)])
        optz_disc = tf.train.AdamOptimizer(C.learning_rate, beta1=0.5, beta2=0.9) \
            .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(disc_name)])
    loss = [loss_gene, loss_disc]
    optz = [optz_gene, optz_disc]

    int_ground = tf.cast(out_ground * 255, tf.uint8)  # for buff fetch
    train_fetch = [int_ground, inp_mask01, loss, optz]

    '''eval'''
    eva_fake03 = auto_encoder(out_ground, 32, 1, disc_name, reuse=True) * ten_repeat
    eval_fetch = [ten_ground, out_ground, ten_patch3, ten_mask03, eva_fake03]
    return inp_ground, inp_mask01, inp_grdbuf, inp_mskbuf, train_fetch, eval_fetch


def process_train(feed_queue, buff_queue):
    print("||Training Initialize")
    inp_ground, inp_mask01, inp_grdbuf, inp_mskbuf, fetch, eval_fetch = init_train()

    sess = mod_util.get_sess(C)
    saver, logger, pre_epoch = mod_util.get_saver_logger(C, sess)
    print("||Training Check")
    eval_list = feed_queue.get()
    eval_feed_dict = {inp_ground: eval_list[0],
                      inp_mask01: eval_list[1],
                      inp_grdbuf: eval_list[2],
                      inp_mskbuf: eval_list[3], }
    sess.run(eval_fetch, eval_feed_dict)

    print("||Training Start")
    start_time = show_time = eval_time = time.time()
    try:
        for epoch in range(C.train_epoch):
            batch_losses = list()  # init
            for i in range(C.batch_size):
                batch_data = feed_queue.get()
                batch_dict = {
                    inp_ground: batch_data[0],
                    inp_mask01: batch_data[1],
                    inp_grdbuf: batch_data[2],
                    inp_mskbuf: batch_data[3],
                }
                buf_ground, buf_mask01, loss, optz = sess.run(fetch, batch_dict)
                batch_losses.append(loss)
                buff_queue.put((i * C.batch_epoch, buf_ground, buf_mask01))

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
            if time.time() - eval_time > C.eval_gap:
                eval_time = time.time()
                logger.close()
                logger = open(C.model_log, 'a')

                eval_feed_dict[inp_mask01] = np.rot90(eval_feed_dict[inp_mask01], axes=(1, 2))
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
    print("| SAVE: %s" % C.model_path)
    mod_util.save_npy(sess, C.model_npz)

    logger.close()
    sess.close()

    mod_util.draw_plot(C.model_log)


def process_feed(feed_queue, buff_queue):
    ts = C.train_size
    bs = C.batch_size
    rd = np.random
    rd_randint = rd.randint
    rd_shuffle = rd.shuffle

    def get_mask01(mats, percentile=85):
        percentile += rd_randint(-2, +3)
        # np.median(ary) == np.percentile(ary, 50)
        # np.quantile(ary) == np.percentile(ary, 75)
        # thr = cv2.threshold(img, np.percentile(img, 85), 255, cv2.THRESH_BINARY)[1]
        thresholds = np.percentile(mats, percentile, axis=(1, 2, 3), keepdims=True)
        mats[mats < thresholds] = 0
        mats[mats >= thresholds] = 255
        return mats

    timer = time.time()
    grounds = img_util.get_data__ground(ts, channel=3)
    print("||Data_sets: ready for check. Used time:", int(time.time() - timer))
    mask01s = img_util.get_data__cloud1(ts)
    print("||Data_sets: ready for check. Used time:", int(time.time() - timer))
    grounds_buff = np.copy(grounds)
    print("||Data_sets: ready for check. Used time:", int(time.time() - timer))
    mask01s_buff = np.zeros_like(mask01s)
    print("||Data_sets: ready for check. Used time:", int(time.time() - timer))
    rd_shuffle(grounds)
    print("||Data_sets: ready for check. Used time:", int(time.time() - timer))

    eval_id = np.random.randint(ts // 2, ts, C.eval_size * 4)
    eval_id = list(set(eval_id))[:C.eval_size]
    feed_queue.put([grounds[eval_id],
                    get_mask01(mask01s[eval_id]),
                    grounds_buff[eval_id],
                    mask01s_buff[eval_id], ])  # for eval

    print("||Data_sets: ready for training")
    i0_range = np.arange(C.batch_epoch)
    i1_range = np.arange(C.batch_epoch)
    replace_ids = np.arange(C.batch_size)
    try:
        for epoch in range(C.train_epoch):
            rd_shuffle(i0_range)
            rd_shuffle(i1_range)
            for i0, i1 in zip(i0_range, i1_range):
                j = i0 * bs

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
                                get_mask01(mask01s[j:j + bs]),
                                grounds_buff[j:j + bs],
                                mask01s_buff[j:j + bs], ])

                while buff_queue.qsize() > 0:
                    k, grounds_get, cloud1s_get = buff_queue.get()

                    rd_shuffle(replace_ids)
                    for replace_id in replace_ids[:C.replace_num]:
                        grounds_buff[replace_id + k] = grounds_get[replace_id]
                        mask01s_buff[replace_id + k] = cloud1s_get[replace_id]

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
    buff_queue = mp.Queue(maxsize=8)
    process = [mp.Process(target=process_feed, args=(feed_queue, buff_queue)),
               mp.Process(target=process_train, args=(feed_queue, buff_queue)), ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run()
