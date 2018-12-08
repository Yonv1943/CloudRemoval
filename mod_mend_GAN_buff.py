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
"""


class Config(object):
    train_epoch = 2 ** 14
    train_size = int(2 ** 17 * 1.9)
    eval_size = 2 ** 3
    batch_size = 2 ** 4
    batch_epoch = train_size // batch_size

    size = int(2 ** 7)
    replace_num = int(0.25 * batch_size)

    show_gap = 2 ** 5  # time
    eval_gap = 2 ** 9  # time
    gpu_limit = 0.48  # 0.0 ~ 1.0
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

    print("|| TEST")

C = Config('mod_mend_GAN_buff')
tf.set_random_seed(1943)


def auto_encoder(inp0, dim, out_dim, name, reuse):
    def leru_batch_norm(ten):
        ten = tf.layers.batch_normalization(ten, training=True)
        ten = tf.nn.leaky_relu(ten)
        return ten

    def conv_tp_conv(ten, idx, step0=1, step1=2):
        filters = (2 ** idx) * dim
        ten = tl.conv2d_transpose(ten, filters, 3, step0, 'valid', activation=leru_batch_norm)
        ten = tl.conv2d(ten, filters, 3, step1, 'valid', activation=tf.nn.leaky_relu)
        return ten

    def conv_tp_conv_tp(ten, idx, step0=2, step1=1):
        filters = (2 ** idx) * dim
        ten = tl.conv2d_transpose(ten, filters, 3, step0, 'valid', activation=leru_batch_norm)
        ten = tl.conv2d(ten, filters, 3, step1, 'same', activation=tf.nn.leaky_relu)
        return ten[:, :-1, :-1, :]

    with tf.variable_scope(name, reuse=reuse):
        ten1 = conv_tp_conv(inp0, 0, 1, 2)
        ten2 = conv_tp_conv(ten1, 1, 1, 2)
        ten3 = conv_tp_conv(ten2, 2, 1, 2)
        ten4 = conv_tp_conv(ten3, 3, 1, 2)
        ten5 = conv_tp_conv(ten4, 4, 1, 2)

        ten6 = conv_tp_conv(ten5, 4, 1, 1) + ten5
        ten7 = conv_tp_conv(ten6, 4, 1, 1) + ten6
        ten8 = conv_tp_conv(ten7, 4, 1, 1) + ten7

        ten4 = conv_tp_conv_tp(ten8, 4, 2, 1)
        ten3 = conv_tp_conv_tp(ten4, 3, 2, 1)
        ten2 = conv_tp_conv_tp(ten3, 2, 2, 1)
        ten1 = conv_tp_conv_tp(ten2, 1, 2, 1)
        ten0 = conv_tp_conv_tp(ten1, 0, 2, 1)

        ten0 = conv_tp_conv(ten0, 0, 1, 1)
        ten0 = tl.conv2d(ten0, out_dim, 1, 1, 'same', activation=tf.nn.tanh)
        return ten0 * 0.505 + 0.5


def process_train(feed_queue, buff_queue):
    # tf.reset_default_graph()
    gene_name = 'gene'
    disc_name = 'disc'
    resize = (C.size // 4, C.size // 4)
    resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

    print("||Training Initialize")
    inp_ground = tf.placeholder(tf.uint8, [None, C.size, C.size, 3])
    inp_cloud1 = tf.placeholder(tf.uint8, [None, C.size, C.size, 1])
    flt_ground = tf.to_float(inp_ground) / 255.0
    flt_cloud1 = tf.to_float(inp_cloud1) / 255.0
    ten_repeat = tf.ones([1, 1, 1, 3])

    ten_ground = flt_ground[:C.batch_size]
    buf_ground = flt_ground[C.batch_size:]
    ten_cloud1 = flt_cloud1[:C.batch_size]

    ten_cloud3 = ten_cloud1 * ten_repeat
    ten_mask10 = (1.0 - ten_cloud3)
    ten_ragged = ten_ground * ten_mask10

    ten_patch3 = auto_encoder(tf.concat((ten_ragged, ten_cloud3), axis=3),
                              32, 3, gene_name, reuse=False)
    out_ground = ten_ragged + ten_patch3 * ten_cloud3
    int_ground = tf.cast(out_ground * 255, tf.uint8)  # for buff fetch

    disc_refabu = auto_encoder(tf.concat((ten_ground, out_ground, buf_ground), axis=0),
                               32, 1, disc_name, reuse=False)
    disc_fake = disc_refabu[C.batch_size:C.batch_size * 2]
    disc_fake3 = disc_fake * ten_repeat
    zero_label = tf.zeros_like(disc_fake)

    loss_gene = tf.losses.mean_pairwise_squared_error(disc_fake, zero_label) + \
                tf.losses.mean_pairwise_squared_error(tf.image.resize_images(ten_ground, resize, resize_method),
                                                      tf.image.resize_images(ten_patch3, resize, resize_method), )

    loss_disc = tf.losses.mean_pairwise_squared_error(disc_refabu,
                                                      tf.concat((zero_label, flt_cloud1), axis=0))

    tf_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optz_gene = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_gene, var_list=[v for v in tf_vars if v.name.startswith(gene_name)])
        optz_disc = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(disc_name)])
    loss = [loss_gene, loss_disc]
    optz = [optz_gene, optz_disc]

    sess = mod_util.get_sess(C)
    saver, logger, pre_epoch = mod_util.get_saver_logger(C, sess)
    print("||Training Check")
    eval_list = feed_queue.get()
    eval_fetch = [ten_ground, out_ground, ten_patch3, ten_cloud3, disc_fake3]
    eval_feed_dict = {inp_ground: eval_list[0],
                      inp_cloud1: eval_list[1]}
    sess.run(eval_fetch, eval_feed_dict)
    sess.run([loss, optz], {inp_ground: eval_list[0],
                            inp_cloud1: eval_list[1]})

    print("||Training Start")
    start_time = show_time = eval_time = time.time()
    try:
        for epoch in range(C.train_epoch):
            batch_losses = list()  # init
            for i in range(C.batch_size):
                feed_data = feed_queue.get()
                batch_return = sess.run((int_ground, loss, optz),
                                        {inp_ground: feed_data[0], inp_cloud1: feed_data[1]})
                batch_losses.append(batch_return[1])
                buff_queue.put((i * C.batch_epoch, batch_return[0], feed_data[1][:C.batch_size]))

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


def process_feed(feed_queue, buff_queue):
    ts = C.train_size
    bs = C.batch_size
    rd_randint = np.random.randint

    def get_mask01(cloud1):
        threshold = rd_randint(160, 192)

        int_cloud1 = cloud1
        int_cloud1[int_cloud1 < threshold] = 0
        int_cloud1[int_cloud1 >= threshold] = 255
        return int_cloud1

    timer = time.time()
    grounds = img_util.get_data__ground(ts, channel=3)
    cloud1s = img_util.get_data__cloud1(ts)
    grounds_buff = np.copy(grounds)
    cloud1s_buff = np.copy(cloud1s)

    print("  load data: %d sec" % (time.time() - timer))

    print("||Data_sets: ready for check. Used time:", int(time.time() - timer))
    eval_id = np.random.randint(ts // 2, ts, C.eval_size * 4)
    # eval_id = list(set(eval_id))[:C.eval_size]
    eval_id = list(set(eval_id))[:C.eval_size * 2]
    feed_queue.put([grounds[eval_id], get_mask01(cloud1s[eval_id])])  # for eval

    print("||Data_sets: ready for training")
    i0_range = np.arange(C.batch_epoch)
    i1_range = np.arange(C.batch_epoch)
    try:
        for epoch in range(C.train_epoch):
            np.random.shuffle(i0_range)
            np.random.shuffle(i1_range)
            for i0, i1 in zip(i0_range, i1_range):
                j = i0 * bs

                int_grounds = np.concatenate((grounds[j:j + bs], grounds_buff[j:j + bs]), axis=0)
                int_cloud1s = np.concatenate((get_mask01(cloud1s[j:j + bs]), cloud1s_buff[j:j + bs]), axis=0)
                feed_queue.put([int_grounds, int_cloud1s])

                while buff_queue.qsize() > 0:
                    k, grounds_get, cloud1s_get = buff_queue.get()
                    replace_idxs = rd_randint(0, C.batch_size, C.replace_num)
                    for replace_idx in replace_idxs:
                        grounds_buff[replace_idx + k] = grounds_get[replace_idx]
                        cloud1s_buff[replace_idx + k] = cloud1s_get[replace_idx]


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
