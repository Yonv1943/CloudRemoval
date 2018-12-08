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

'''
Reference: https://github.com/jiamings/wgan
Reference: https://github.com/cameronfabbri/Improved-Wasserstein-GAN
Reference: https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN

2018-10-10 Modify: Yonv1943

2018-10-11 save eval jpg
2018-10-12 'TF_CPP_MIN_LOG_LEVEL' tf.Session()
2018-10-12 origin, tensorflow.contrib.layers --> tf.layers
2018-10-12 elegant coding, stable
2018-10-13 C.size  28 --> 32, deeper, dcgan
2018-10-15 cloud removal
2018-10-21 'class Tools' move from mod_*.py to util.img_util.py 
2018-10-22 change mask from 'middle square' to 'spot'
2018-10-23 spot --> polygon
'''

C = Config('mod_GAN_poly')
T = img_util.Tools()
rd.seed(1943)


def model_save_npy(sess, print_info):
    tf_vars = tf.global_variables()

    '''save as singal npy'''
    npy_dict = dict()
    for var in tf_vars:
        npy_dict[var.name] = var.eval(session=sess)
        print("| FETCH: %s" % var.name) if print_info else None
    np.savez(C.model_npz, npy_dict)
    with open(C.model_npz + '.txt', 'w') as f:
        f.writelines(["%s\n" % key for key in npy_dict.keys()])

    '''save as several npy'''
    # shutil.rmtree(C.model_npy, ignore_errors=True)
    # os.makedirs(C.model_npy, exist_ok=True)
    # for v in tf_vars:
    #     v_name = str(v.name).replace('/', '-').replace(':', '.') + '.npy'
    #     np.save(os.path.join(C.model_npy, v_name), v.eval(session=sess))
    #     print("| SAVE %s.npy" % v.name) if print_info else None
    print("| SAVE: %s" % C.model_npz)


def model_load_npy(sess):
    tf_dict = dict()
    for tf_var in tf.global_variables():
        tf_dict[tf_var.name] = tf_var

    for npy_name in os.listdir(C.model_npz):
        var_name = npy_name[:-4].replace('-', '/').replace('.', ':')
        var_node = tf_dict.get(var_name, None)
        if npy_name.find('defog') == 0 and var_node:
            var_ary = np.load(os.path.join(C.model_npz, npy_name))
            sess.run(tf.assign(var_node, var_ary)) if var_node else None

    if os.path.exists(C.model_npz):
        # sess.run(tf.global_variables_initializer())
        # sess.run([i.assign(np.load(G.model_npz)[i.name]) for i in tf.trainable_variables()])
        sess.run([i.assign(np.load(C.model_npz)[i.name]) for i in tf.global_variables()])
        print("| Load from npz:", C.model_path)


def sess_saver_logger():
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=C.gpu_limit))
    config.gpu_options.allow_growth = True

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore the TensorFlow log messages
    sess = tf.Session(config=config)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # recover the TensorFlow log messages

    saver = tf.train.Saver(max_to_keep=4)
    # if os.path.exists(os.path.join(C.model_dir, 'checkpoint')):
    try:
        # C.model_path = tf.train.latest_checkpoint(C.model_dir)
        saver.restore(sess, C.model_path)
        print("| Load from checkpoint:", C.model_path)
    except Exception:
        os.makedirs(C.model_dir, exist_ok=True)
        sess.run(tf.global_variables_initializer())
        print("| Init:", C.model_path)

        if os.path.exists(C.model_npz):
            print("| Load from mod.npz")
            name2ary = np.load(C.model_npz)
            for var_node in tf.global_variables():
                try:
                    if str(var_node.name).find('defop') == 0:
                        var_ary = name2ary[var_node.name]
                        sess.run(tf.assign(var_node, var_ary))
                except:
                    pass

    logger = open(C.model_log, 'a')
    previous_train_epoch = sum(1 for _ in open(C.model_log)) if os.path.exists(C.model_log) else 0
    print('| Train_epoch: %6d+%6d' % (previous_train_epoch, C.train_epoch))
    print('| Batch_epoch: %6dx%6d' % (C.batch_epoch, C.batch_size))
    return sess, saver, logger, previous_train_epoch


def leRU_batch_norm(ten):
    ten = tf.layers.batch_normalization(ten, training=True)
    ten = tf.nn.leaky_relu(ten)
    return ten


def generator(inp0, dim, name, reuse):
    # inp0 = tf.placeholder(tf.float32, [None, G.size, G.size, 1])  # G.size == 2 ** 8
    with tf.variable_scope(name, reuse=reuse):
        ten1 = tl.conv2d(inp0, 1 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten2 = tl.conv2d(ten1, 2 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten3 = tl.conv2d(ten2, 4 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten4 = tl.conv2d(ten3, 8 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten5 = tl.conv2d(ten4, 16 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)

        ten6 = tl.conv2d(ten5, 16 * dim, 3, 1, 'same', activation=tf.nn.leaky_relu)
        ten6 = tl.conv2d(ten6, 16 * dim, 3, 1, 'same', activation=tf.nn.leaky_relu)
        ten6 = tl.conv2d(ten6, 16 * dim, 3, 1, 'same', activation=tf.nn.leaky_relu)
        ten6 = tl.conv2d(ten6, 16 * dim, 3, 1, 'same', activation=tf.nn.leaky_relu)
        ten6 = tl.conv2d_transpose(ten6, 16 * dim, 3, 1, 'same', activation=leRU_batch_norm)
        ten6 = tl.conv2d_transpose(ten6, 16 * dim, 3, 1, 'same', activation=leRU_batch_norm)
        ten6 = tl.conv2d_transpose(ten6, 16 * dim, 3, 1, 'same', activation=leRU_batch_norm)
        ten6 = tl.conv2d_transpose(ten6, 16 * dim, 3, 1, 'same', activation=leRU_batch_norm)

        ten5 = tl.conv2d_transpose(ten6, 16 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten4 = tl.conv2d_transpose(ten5, 8 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten3 = tl.conv2d_transpose(ten4, 4 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten2 = tl.conv2d_transpose(ten3, 2 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten1 = tl.conv2d_transpose(ten2, 1 * dim, 4, 2, 'same', activation=leRU_batch_norm)

        ten1 = tf.concat((ten1, inp0), axis=3)
        ten1 = tl.conv2d(ten1, 1 * dim, 3, 1, 'same', activation=leRU_batch_norm)
        ten1 = tl.conv2d(ten1, 3, 3, 1, 'same', activation=tf.nn.tanh)
        ten1 = ten1 * 0.505 + 0.5
        return ten1


def discriminator(ten0, ten1, dim, name, reuse=True):
    # ten0 = tf.placeholder(tf.float32, [None, C.size, C.size, 3])  # ragged
    # ten1 = tf.placeholder(tf.float32, [None, C.size, C.size, 3])  # patch3
    with tf.variable_scope(name, reuse=reuse):
        ten0 = tl.conv2d(ten0, dim * 1, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten0 = tl.conv2d(ten0, dim * 2, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten0 = tl.conv2d(ten0, dim * 4, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten0 = tl.conv2d(ten0, dim * 8, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten0 = tl.conv2d(ten0, dim * 16, 4, 2, 'same', activation=tf.nn.leaky_relu)

        ten1 = tl.conv2d(ten1, dim * 1, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten1 = tl.conv2d(ten1, dim * 2, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten1 = tl.conv2d(ten1, dim * 4, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten1 = tl.conv2d(ten1, dim * 8, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten1 = tl.conv2d(ten1, dim * 16, 4, 2, 'same', activation=tf.nn.leaky_relu)

        ten = tf.concat([ten0, ten1], axis=3)
        ten = tl.flatten(ten)
        ten = tl.dense(ten, 1 * dim, activation=tf.nn.leaky_relu)
        ten = tl.dense(ten, 1)
        return ten


def process_train(feed_queue, buff_queue):  # model_train
    gene_name = 'gene'
    disc_name = 'disc'

    inp_aerial = tf.placeholder(tf.float32, [None, C.size, C.size, 3])
    inp_buffer = tf.placeholder(tf.float32, [None, C.size, C.size, 3])
    inp_cloud1 = tf.placeholder(tf.float32, [None, C.size, C.size, 1])
    # batch_size = tf.shape(inp_aerial)[0]
    # ten_mask01 = tf.ones([batch_size, C.size, C.size, 3])

    # thick = tf.random_uniform([], dtype=tf.float32) * 0.25 + 0.5
    # ten_cloud1 = tf.clip_by_value((inp_cloud1 - thick) * 8.0, 0.0, 1.0)
    ten_mask01 = inp_cloud1 * tf.ones([1, 1, 1, 3], tf.float32)
    ten_mask10 = 1.0 - ten_mask01

    ten_ragged = inp_aerial * ten_mask10
    ten_aerial = tf.concat([ten_ragged, inp_cloud1], axis=3)

    ten_patch1 = generator(ten_aerial, dim=16, name=gene_name, reuse=False) * ten_mask01
    out_aerial = ten_ragged + ten_patch1

    '''loss, optz'''
    real_disc = discriminator(inp_aerial, inp_aerial * ten_mask01, 16, disc_name, reuse=False)
    buff_disc = discriminator(inp_buffer, inp_buffer * ten_mask01, 16, disc_name, reuse=True)
    fake_disc = discriminator(out_aerial, ten_patch1, 16, disc_name, reuse=True)

    label_ones = tf.ones([C.batch_size, 1])
    label_zero = tf.zeros([C.batch_size, 1])

    loss_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_disc, labels=label_ones, ))

    loss_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=real_disc, labels=label_ones, ))
    loss_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_disc, labels=label_zero, ))
    loss_disc_buff = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=buff_disc, labels=label_zero, ))
    loss_disc = loss_disc_real + (loss_disc_fake + loss_disc_buff) * 0.5

    tf_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optz_gene = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_gene, var_list=[v for v in tf_vars if v.name.startswith(gene_name)])
        optz_disc = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(disc_name)])
        # optz_gene = tf.train.RMSPropOptimizer(learning_rate=1e-5) \
        #     .minimize(loss_gene, var_list=[v for v in tf_vars if v.name.startswith(gene_name)])
        # optz_disc = tf.train.RMSPropOptimizer(learning_rate=1e-5) \
        #     .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(disc_name)])

    loss = [loss_gene, loss_disc]
    optz = [optz_gene, optz_disc]

    """model train"""
    sess, saver, logger, previous_train_epoch = sess_saver_logger()
    feed_dict = dict()
    eval_out_fetch = [inp_aerial, out_aerial, ten_patch1, ten_mask01]
    eval_feed_list = feed_queue.get()
    eval_feed_dict = {inp_aerial: eval_feed_list[0],
                      inp_cloud1: eval_feed_list[1], }

    '''model check'''
    print("||Training Check")
    feed_list = feed_queue.get()
    feed_dict[inp_aerial] = feed_list[0]
    feed_dict[inp_buffer] = feed_list[1]
    feed_dict[inp_cloud1] = feed_list[2]
    loss_average = sess.run([loss, optz], feed_dict)[0]

    epoch = 0
    start_time = show_time = save_time = time.time()
    print("||Training Start")
    try:
        for epoch in range(C.train_epoch):
            l_gene, l_disc = loss_average
            disable_gene = bool(l_gene * C.active_rate < l_disc)
            disable_disc = bool(l_disc * C.active_rate < l_gene)

            if disable_disc:
                optz = optz_gene
            elif disable_gene:
                optz = optz_disc
            else:
                optz = [optz_gene, optz_disc]

            batch_losses = list()  # init
            for i in range(C.batch_epoch):
                feed_dict[inp_aerial], feed_dict[inp_buffer], feed_dict[inp_cloud1] = feed_queue.get()
                buff_put, batch_loss, _ = sess.run([ten_patch1, loss, optz], feed_dict)

                batch_losses.append(batch_loss)
                buff_queue.put((i * C.batch_size, buff_put))

            loss_average = np.mean(batch_losses, axis=0)
            loss_error = np.std(batch_losses, axis=0)

            logger.write('%e %e %e %e\n'
                         % (loss_average[0], loss_error[0],
                            loss_average[1], loss_error[1],))

            if time.time() - show_time > C.show_gap:
                show_time = time.time()
                remain_epoch = C.train_epoch - epoch
                remain_time = (show_time - start_time) * remain_epoch / (epoch + 1)
                print(end="\n|  %3d s |%3d epoch | Loss: %9.3e %9.3e | G-%d D-%d"
                          % (remain_time, remain_epoch,
                             loss_average[0], loss_average[1],
                             not disable_gene, not disable_disc))
            if time.time() - save_time > C.save_gap:
                '''save model'''
                save_time = time.time()
                # saver.save(sess, C.model_path, write_meta_graph=False)
                logger.close()
                logger = open(C.model_log, 'a')
                print(end="\n||SAVE")

                T.eval_and_get_img(mat_list=sess.run(eval_out_fetch, eval_feed_dict), channel=3,
                                   img_path=os.path.join(C.model_dir, "eval-%08d.jpg"
                                                         % (previous_train_epoch + epoch)), )

        '''save model'''
        saver.save(sess, C.model_path, write_meta_graph=False)
        print("| SAVE: %s" % C.model_path)

        T.eval_and_get_img(mat_list=sess.run(eval_out_fetch, eval_feed_dict), channel=3,
                           img_path=os.path.join(C.model_dir, "eval-%08d.jpg"
                                                 % (previous_train_epoch + epoch)))
    except KeyboardInterrupt:
        print("| KeyboardInterrupt")
        saver.save(sess, C.model_path, write_meta_graph=False)
        print("| SAVE: %s" % C.model_path)
        model_save_npy(sess, print_info=False)
        print("| SAVE: %s" % C.model_npz)

    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    print('| Train_epoch: %d' % C.train_epoch)
    print('| TimeUsed:    %d' % int(time.time() - start_time))
    logger.close()
    sess.close()

    T.draw_plot(C.model_log)


def process_data(feed_queue, buff_queue):
    ts = C.train_size
    bs = C.batch_size

    data_aerial = img_util.get_data__aerial(ts, channel=3)
    data_buffer = img_util.get_data__buffer(ts, channel=3)
    data_cloud1 = img_util.get_data__poly01(bs)
    print("||Data_sets: ready for check")
    eval_id = rd.randint(ts // 2, ts, C.eval_size * 2)
    eval_id = list(set(eval_id))[:C.eval_size]
    feed_queue.put([data_aerial[eval_id],
                    data_cloud1[:C.eval_size]])  # for eval
    feed_queue.put([data_aerial[:bs],
                    data_buffer[:bs],
                    data_cloud1[:bs], ])  # for check

    try:
        print("||Data_sets: ready for training")
        for epoch in range(C.train_epoch):
            if epoch % 2 == 0:  # refresh cloud1
                data_cloud1 = img_util.get_data__poly01(bs)
            elif epoch % 8 == 1:  # np.rot90()
                data_aerial[:bs] = data_aerial[:bs].transpose((0, 2, 1, 3))
                data_buffer[:bs] = data_buffer[:bs].transpose((0, 2, 1, 3))
            else:  # shuffle mildly
                rd_j, rd_k = rd.randint(0, int(ts * 0.5 - bs), size=2)
                rd_k += int(ts * 0.5)

                data_aerial[rd_j:rd_j + bs], data_aerial[rd_k:rd_k + bs] = \
                    data_aerial[rd_k:rd_k + bs], data_aerial[rd_j:rd_j + bs]
                data_buffer[rd_j:rd_j + bs], data_buffer[rd_k:rd_k + bs] = \
                    data_buffer[rd_k:rd_k + bs], data_buffer[rd_j:rd_j + bs]

            for i in range(C.batch_epoch):
                j = i * bs
                feed_queue.put([
                    data_aerial[j: j + bs],
                    data_buffer[j: j + bs],
                    data_cloud1,
                ])

                while buff_queue.qsize() > 0:
                    k, buff_get = buff_queue.get()
                    replace_idxs = rd.randint(0, bs, C.replace_num)
                    for replace_idx in replace_idxs:
                        data_buffer[replace_idx + k] = buff_get[replace_idx]

                # cv2.imshow('', data_aerial[0])
                # cv2.waitKey(234)

    except KeyboardInterrupt:
        print("| quit:", process_data.__name__)


def run():  # beta
    # T.draw_plot(C.model_log)
    if input("||PRESS 'y' to REMOVE model_dir? %s : " % C.model_dir) == 'y':
        shutil.rmtree(C.model_dir, ignore_errors=True)
        print("||REMOVE")
    else:
        print("||KEEP")

    import multiprocessing as mp
    feed_queue = mp.Queue(maxsize=8)
    buff_queue = mp.Queue(maxsize=8)

    process = [mp.Process(target=process_data, args=(feed_queue, buff_queue)),
               mp.Process(target=process_train, args=(feed_queue, buff_queue)), ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run()
