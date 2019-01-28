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
2018-12-29 stable
"""


class Config(object):
    train_epoch = int(2 ** 13 * 1.5)
    train_size = int(2 ** 17 * 1.2)
    eval_size = 2 ** 4 - 2  # 2 ** 3
    batch_size = int(2 ** 4)
    batch_epoch = train_size // batch_size

    size = int(2 ** 8)
    replace_num = int(0.368 * batch_size)
    learning_rate = 8e-5  # 1e-4

    show_gap = 2 ** 7  # time
    eval_gap = 2 ** 11  # time
    gpu_limit = 0.9  # 0.0 ~ 1.0
    gpu_id = 0

    data_dir = '/mnt/sdb1/data_sets'
    aerial_dir = os.path.join(data_dir, 'AerialImageDataset/train')
    cloud_dir = os.path.join(data_dir, 'ftp.nnvl.noaa.gov_color_IR_2018')
    grey_dir = os.path.join(data_dir, 'CloudGreyDataset_%dx%d' % (size, size))

    def __init__(self, model_dir='mod'):
        self.model_dir = model_dir
        self.model_name = 'mod'
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.model_npz = os.path.join(self.model_dir, self.model_name + '.npz')
        self.model_log = os.path.join(self.model_dir, 'training_npy.txt')


if __name__ != '__main__':
    from configure import Config  # for test

    print("|| TEST")

C = Config('mod_cloud_remove_rec')
tf.set_random_seed(time.time() * 1943 % 178320049)


def auto_encoder(inp0, dim, out_dim, name, reuse, training=True):
    padding1 = tf.constant(((0, 0), (1, 1), (1, 1), (0, 0)))

    def leru_batch_norm(ten):
        ten = tl.batch_normalization(ten, training=training)
        ten = tf.nn.leaky_relu(ten)
        return ten

    def conv_tp(ten, idx):
        filters = (2 ** idx) * dim
        ten = tl.conv2d_transpose(ten, filters, 3, 2, 'same', activation=leru_batch_norm)
        return ten

    def conv_pad1(ten, idx, step=1):
        filters = (2 ** idx) * dim
        ten = tf.pad(ten, padding1, 'REFLECT')
        ten = tl.conv2d(ten, filters, 3, step, 'valid', activation=tf.nn.leaky_relu)
        return ten

    def conv_res(ten, idx):
        return conv_pad1(conv_pad1(ten, idx), idx) + conv_pad1(ten, idx)

    with tf.variable_scope(name, reuse=reuse):
        ten1 = conv_pad1(inp0, 0, 2)
        ten2 = conv_pad1(ten1, 1, 2)
        ten3 = conv_pad1(ten2, 2, 2)
        ten4 = conv_pad1(ten3, 3, 2)
        ten5 = conv_pad1(ten4, 4, 2)
        ten6 = conv_pad1(ten5, 5, 2)

        ten6 = conv_res(ten6, 5)

        ten5 = conv_res(ten5, 4)
        ten5 = conv_res(ten5, 4)
        ten5 = tf.concat((ten5, conv_tp(ten6, 5)), axis=3)
        ten5 = conv_pad1(ten5, 5, 1)

        ten4 = conv_res(ten4, 3)
        ten4 = conv_res(ten4, 3)
        ten4 = conv_res(ten4, 3)
        ten4 = tf.concat((ten4, conv_tp(ten5, 4)), axis=3)
        ten4 = conv_pad1(ten4, 4, 1)

        ten3 = conv_tp(ten4, 3)
        ten2 = conv_tp(ten3, 2)
        ten1 = conv_tp(ten2, 1)
        ten0 = conv_tp(ten1, 0)

        ten0 = conv_pad1(ten0, 0, 1)
        ten0 = tf.concat((ten0, inp0), axis=3)
        ten0 = tl.conv2d(ten0, out_dim, 1, 1, 'same', activation=tf.nn.sigmoid)
        return ten0


def init_train():
    # tf.reset_default_graph()
    gene_name, gene_dim = 'gene', 32
    disc_name, disc_dim = 'disc', 32

    '''init'''
    inp_ground = tf.placeholder(tf.uint8, [None, C.size, C.size, 3])
    ten_ground = tf.to_float(inp_ground)
    ten_ground *= tf.random_uniform([], 0.00382, 0.00402) + tf.random_uniform([1, 1, 1, 3], -0.00012, 0.00012)
    ten_ground += tf.random_uniform([], -0.02, 0.02) + tf.random_uniform([1, 1, 1, 3], -0.02, 0.02)
    ten_ground = tf.clip_by_value(ten_ground, 0, 1)

    inp_mask01 = tf.placeholder(tf.uint8, [None, C.size, C.size, 1])
    ten_mask01 = tf.to_float(inp_mask01) / 255

    '''func'''
    ten_mask10 = (1.0 - ten_mask01)
    ten_ragged = ten_ground * ten_mask10

    ten_patch3 = auto_encoder(ten_ragged - ten_mask01,
                              gene_dim, 3, gene_name, reuse=False)
    out_ground = ten_ragged + ten_patch3 * ten_mask01

    dis_real_1 = auto_encoder(ten_ground, disc_dim, 1, disc_name, reuse=False)
    dis_fake_1 = auto_encoder(out_ground, disc_dim, 1, disc_name, reuse=True)

    '''buff'''
    inp_grdbuf = tf.placeholder(tf.uint8, [None, C.size, C.size, 3])
    ten_grdbuf = tf.to_float(inp_grdbuf) / 255

    inp_mskbuf = tf.placeholder(tf.uint8, [None, C.size, C.size, 1])
    ten_mskbuf = tf.to_float(inp_mskbuf) / 255
    dis_buff_1 = auto_encoder(ten_grdbuf, disc_dim, 1, disc_name, reuse=True)

    '''loss'''
    zero_mask1 = tf.zeros_like(dis_real_1)

    # dif_patch3 = ten_ground - ten_patch3
    # dif_patch3 = tf.image.resize_images(dif_patch3, (C.size//4, C.size//4))

    # loss_gene = tf.reduce_mean((zero_mask1 - dis_fake_1) ** 2 * 3)
    # loss_gene += tf.reduce_mean((ten_ground - ten_patch3) ** 2)
    #
    # loss_disc = tf.reduce_mean((zero_mask1 - dis_real_1) ** 2)
    # loss_disc += tf.reduce_mean((ten_mask01 - dis_fake_1) ** 2)
    # loss_disc += tf.reduce_mean((ten_mskbuf - dis_buff_1) ** 2)  # buffer

    loss_gene = tf.losses.mean_pairwise_squared_error(zero_mask1, dis_fake_1) * 2
    loss_gene += tf.losses.mean_pairwise_squared_error(ten_ground * ten_mask01,
                                                       ten_patch3 * ten_mask01)

    loss_disc = tf.losses.mean_pairwise_squared_error(zero_mask1, dis_real_1)
    loss_disc += tf.losses.mean_pairwise_squared_error(ten_mask01, dis_fake_1)
    loss_disc += tf.losses.mean_pairwise_squared_error(ten_mskbuf, dis_buff_1)  # buffer

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        tf_vars = tf.trainable_variables()
        optz_gene = tf.train.AdamOptimizer(C.learning_rate, beta1=0.5, beta2=0.9) \
            .minimize(loss_gene, var_list=[v for v in tf_vars if v.name.startswith(gene_name)])
        optz_disc = tf.train.AdamOptimizer(C.learning_rate, beta1=0.5, beta2=0.9) \
            .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(disc_name)])
        loss = [loss_gene, loss_disc]
        optz = [optz_gene, optz_disc]

    int_ground = tf.cast(out_ground * 255, tf.uint8)  # for buff fetch
    int_mask01 = tf.cast(ten_mask01 * 255, tf.uint8)  # for buff fetch
    train_fetch = [int_ground, int_mask01, loss, optz]

    eval_fetch = [ten_ground, ten_patch3,
                  out_ground, ten_mask01, dis_fake_1]
    return inp_ground, inp_mask01, inp_grdbuf, inp_mskbuf, train_fetch, eval_fetch


def process_train(feed_queue, buff_queue):
    print("||Training Initialize")
    inp_ground, inp_mask01, inp_grdbuf, inp_mskbuf, fetch, eval_fetch = init_train()
    optz_gene = fetch[3][0]
    optz_disc = fetch[3][1]
    loss_gene = loss_disc = 0

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

                idx = batch_data[0]
                batch_dict = {
                    inp_ground: batch_data[1],
                    inp_mask01: batch_data[2],
                    inp_grdbuf: batch_data[3],
                    inp_mskbuf: batch_data[4],
                }

                # fetch[3] = optz_disc
                if loss_disc * 8 < loss_gene:
                    fetch[3] = optz_gene
                # elif loss_gene * 8 < loss_disc:
                #     fetch[3] = optz_disc
                else:
                    fetch[3] = (optz_gene, optz_disc)

                buf_ground, buf_mask01, (loss_gene, loss_disc), optz = sess.run(fetch, batch_dict)
                batch_losses.append((loss_gene, loss_disc))
                buff_queue.put((idx, buf_ground, buf_mask01))

            loss_average = np.mean(batch_losses, axis=0)
            logger.write('%e %e\n' % (loss_average[0], loss_average[1]))

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
    grounds = img_util.get_data__ground(ts)
    print("  Dataset grounds. Used time:", int(time.time() - timer))
    np.random.shuffle(grounds)
    print("  Dataset shuffle. Used time:", int(time.time() - timer))
    grounds_buff = np.copy(grounds)
    print("  Dataset buffers. Used time:", int(time.time() - timer))
    mask01s = img_util.get_data__cloud1(ts)
    print("  Dataset mask01s. Used time:", int(time.time() - timer))
    mask01s_buff = np.zeros_like(mask01s)
    print("  Dataset buffers. Used time:", int(time.time() - timer))

    eval_id = list(set(np.random.randint(0, ts, C.eval_size * 4)))[:C.eval_size]
    feed_queue.put([grounds[eval_id],
                    get_mask01(mask01s[eval_id]),
                    grounds_buff[eval_id],
                    mask01s_buff[eval_id], ])  # for eval

    print("||Data_sets: ready for training")
    i0_range = np.arange(C.batch_epoch)
    i1_range = np.arange(C.batch_epoch)
    replace_ids = np.arange(C.batch_size)

    def batch_op0(j, k):
        j *= bs
        k *= bs

        mask01_buff = (1 - get_mask01(mask01s[k:k + bs] // 255)).astype(np.uint8)
        ground_buff = grounds[j:j + bs] * mask01_buff

        feed_queue.put([j,
                        grounds[j:j + bs],
                        get_mask01(mask01s[k:k + bs]),
                        ground_buff,
                        mask01_buff, ])

        while buff_queue.qsize() > 0:
            idx, grounds_get, cloud1s_get = buff_queue.get()
            grounds_buff[idx:idx + bs] = grounds_get
            mask01s_buff[idx:idx + bs] = cloud1s_get

    def batch_opn(j, k):
        j *= bs
        k *= bs
        q = rd_randint(ts // 2 - bs)

        switch = rd_randint(6)
        if switch == 0:
            grounds[j:j + bs] = np.rot90(grounds[j:j + bs], axes=(1, 2))
        elif switch == 1:
            grounds[j:j + bs] = np.flip(grounds[j:j + bs], axis=rd_randint(1, 3))
        elif switch == 2:
            mask01s[j:j + bs] = np.rot90(mask01s[j:j + bs], axes=(1, 2))
        elif switch == 3:
            mask01s[j:j + bs] = np.flip(mask01s[j:j + bs], axis=rd_randint(1, 3))
        elif switch == 4 and not j <= q <= j + bs:
            grounds[j:j + bs], grounds[q:q + bs] = grounds[q:q + bs], grounds[j:j + bs]
        elif switch == 5 and not j <= q <= j + bs:
            mask01s[j:j + bs], mask01s[q:q + bs] = mask01s[q:q + bs], mask01s[j:j + bs]

        feed_queue.put([j,
                        grounds[j:j + bs],
                        get_mask01(mask01s[k:k + bs]),
                        grounds_buff[j:j + bs],
                        mask01s_buff[j:j + bs], ])

        while buff_queue.qsize() > 0:
            idx, grounds_get, cloud1s_get = buff_queue.get()
            # grounds_buff[idx:idx+bs] = grounds_get
            # mask01s_buff[idx:idx+bs] = cloud1s_get

            rd_shuffle(replace_ids)
            for replace_id in replace_ids[:C.replace_num]:
                grounds_buff[replace_id + idx] = grounds_get[replace_id]
                mask01s_buff[replace_id + idx] = cloud1s_get[replace_id]

    for _ in (0,):
        for i in range(C.batch_epoch):
            batch_op0(i, i)
    for _ in range(1, C.train_epoch):
        rd_shuffle(i0_range)
        rd_shuffle(i1_range)
        for i0, i1 in zip(i0_range, i1_range):
            batch_opn(i0, i1)


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
    buff_queue = mp.Queue(maxsize=8)
    process = [mp.Process(target=process_feed, args=(feed_queue, buff_queue)),
               mp.Process(target=process_train, args=(feed_queue, buff_queue)), ]

    os.makedirs(os.path.join(C.model_dir, 'TRAINING.MARK'), exist_ok=True)
    [p.start() for p in process]

    # [p.join() for p in process]
    while os.path.exists(os.path.join(C.model_dir, 'TRAINING.MARK')):
        time.sleep(2)
    else:
        [p.terminate() for p in process]


if __name__ == '__main__':
    run()
