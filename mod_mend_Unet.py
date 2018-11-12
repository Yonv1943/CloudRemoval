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
2018-10-23 for discriminator, tf.concat([tenx, mask], axis=0)
2018-11-05 for generator
2018-11-06 for process_data, tiny, U-net for generator
2018-11-06 buffer mask01 for discriminator, inp_dim_4
'''

C = Config('mod_mend_Unet')
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


def conv_mask(ten, mask, dim, idx):
    filters = (2 ** idx) * dim
    size = C.size // (2 ** (idx + 1))
    mask = tf.image.resize_images(mask, (size, size))

    ten = tl.conv2d(ten, filters, 4, 2, 'same', activation=tf.nn.leaky_relu)
    ten = tf.concat((ten, mask), axis=3)
    return ten


def conv_tp_res(inp, dim, idx):
    filters = (2 ** idx) * dim + 1  # add 1 for channel mask

    ten = tl.conv2d(inp, filters, 3, 1, 'valid', activation=tf.nn.leaky_relu)
    ten = tl.conv2d_transpose(ten, filters, 3, 1, 'valid', activation=leRU_batch_norm)
    return ten + inp


def conv_tp_concat(ten, con, dim, idx):
    filters = (2 ** idx) * dim

    ten = tf.concat((ten, con), axis=3)
    ten = tl.conv2d_transpose(ten, filters, 4, 2, 'same', activation=leRU_batch_norm)
    return ten


def generator(inp0, mask, dim, name, reuse):
    # inp0 = tf.placeholder(tf.float32, [None, G.size, G.size, 3])  # G.size > 2 ** 5 + 1
    with tf.variable_scope(name, reuse=reuse):
        inp0 = tf.concat((inp0, mask), axis=3)

        ten1 = conv_mask(inp0, mask, dim, 0)
        ten2 = conv_mask(ten1, mask, dim, 1)
        ten3 = conv_mask(ten2, mask, dim, 2)
        ten4 = conv_mask(ten3, mask, dim, 3)
        ten5 = conv_mask(ten4, mask, dim, 4)

        ten6 = conv_tp_res(ten5, dim, 4)
        ten6 = conv_tp_res(ten6, dim, 4)
        ten6 = conv_tp_res(ten6, dim, 4)
        ten6 = conv_tp_res(ten6, dim, 4)

        ten5 = conv_tp_concat(ten6, ten5, dim, 5)
        ten4 = conv_tp_concat(ten5, ten4, dim, 4)
        ten3 = conv_tp_concat(ten4, ten3, dim, 3)
        ten2 = conv_tp_concat(ten3, ten2, dim, 2)
        ten1 = conv_tp_concat(ten2, ten1, dim, 1)

        ten1 = tf.concat((ten1, inp0), axis=3)
        ten1 = tl.conv2d_transpose(ten1, dim, 3, 1, 'valid', activation=leRU_batch_norm)
        ten1 = tl.conv2d(ten1, 3, 3, 1, 'valid', activation=tf.nn.tanh)
        ten1 = ten1 * 0.505 + 0.5
        return ten1


def discriminator_error(ten0, ten1, mask, dim, name, reuse=True):
    # ten0 = tf.placeholder(tf.float32, [None, C.size, C.size, 3])  # ragged
    # ten1 = tf.placeholder(tf.float32, [None, C.size, C.size, 3])  # patch3
    # mask = tf.placeholder(tf.float32, [None, C.size, C.size, 1])
    with tf.variable_scope(name, reuse=reuse):
        ten0 = tf.concat((ten0, mask), axis=3)
        ten0 = conv_mask(ten0, mask, dim, 0)
        ten0 = conv_mask(ten0, mask, dim, 1)
        ten0 = conv_mask(ten0, mask, dim, 2)
        ten0 = conv_mask(ten0, mask, dim, 3)
        ten0 = conv_mask(ten0, mask, dim, 4)

        ten1 = tf.concat((ten1, mask), axis=3)
        ten1 = conv_mask(ten1, mask, dim, 0)
        ten1 = conv_mask(ten1, mask, dim, 1)
        ten1 = conv_mask(ten1, mask, dim, 2)
        ten1 = conv_mask(ten1, mask, dim, 3)
        ten1 = conv_mask(ten1, mask, dim, 4)

        ten = tf.concat([ten0, ten1], axis=3)
        ten = tl.flatten(ten)
        ten = tl.dense(ten, 1 * dim, activation=tf.nn.leaky_relu)
        ten = tl.dense(ten, 1)
        return ten


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

    inp_aerial = tf.placeholder(tf.float32, [None, C.size, C.size, 4])
    inp_buffer = tf.placeholder(tf.float32, [None, C.size, C.size, 4])
    ten_repeat = tf.constant(np.ones([1, 1, 1, 3]), tf.float32)

    '''gene'''
    ten_aerial = inp_aerial[:, :, :, 0:3]
    ten_mask01 = inp_aerial[:, :, :, 3:4]
    ten_mask03 = ten_mask01 * ten_repeat

    ten_ragged = ten_aerial * (1.0 - ten_mask03)
    ten_patch3 = generator(ten_ragged, ten_mask01, dim=24, name=gene_name, reuse=False)
    out_aerial = ten_ragged + ten_patch3 * ten_mask03

    '''disc'''
    ten_buffer = inp_buffer[:, :, :, 0:3]
    buf_mask01 = inp_buffer[:, :, :, 3:4]
    buf_aerial = ten_ragged + ten_buffer * buf_mask01 * ten_repeat
    buf_return = tf.concat((ten_buffer, buf_mask01), axis=3)

    real_disc = discriminator(ten_aerial, ten_aerial, 24, disc_name, reuse=False)
    fake_disc = discriminator(out_aerial, ten_patch3, 24, disc_name, reuse=True)
    buff_disc = discriminator(buf_aerial, ten_buffer, 24, disc_name, reuse=True)

    '''loss, optz'''
    label_ones = tf.constant(np.ones([C.batch_size, 1]), tf.float32)
    label_zero = tf.constant(np.zeros([C.batch_size, 1]), tf.float32)

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
    eval_out_fetch = [ten_aerial, out_aerial, ten_patch3, ten_mask03]
    eval_feed_list = feed_queue.get()
    eval_feed_dict = {inp_aerial: eval_feed_list[0], }

    '''model check'''
    print("||Training Check")
    feed_list = feed_queue.get()
    feed_dict[inp_aerial] = feed_list[0]
    feed_dict[inp_buffer] = feed_list[1]
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
                feed_dict[inp_aerial], feed_dict[inp_buffer] = feed_queue.get()

                buff_put, batch_loss, _ = sess.run([buf_return, loss, optz], feed_dict)
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
        T.eval_and_get_img(mat_list=sess.run(eval_out_fetch, eval_feed_dict), channel=3,
                           img_path=os.path.join(C.model_dir, "eval-%08d.jpg" % 0))

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
    data_circle = img_util.get_data__circle(ts, circle_num=3)
    data_aerial = np.concatenate((data_aerial, data_circle), axis=3)  # [ts, C.size, C.size, 4]
    del data_circle

    data_buffer = img_util.get_data__buffer(ts, channel=4)  # buf_aerial3 + mask1

    print("||Data_sets: ready for check")
    eval_id = rd.randint(ts // 2, ts, C.eval_size * 2)
    eval_id = list(set(eval_id))[:C.eval_size]
    feed_queue.put([data_aerial[eval_id], ])  # for eval
    feed_queue.put([data_aerial[:bs],
                    data_buffer[:bs], ])  # for check

    print("||Data_sets: ready for training")
    for epoch in range(C.train_epoch):
        if epoch % 8 == 1:  # refresh circle(mask)
            k = rd.randint(ts - bs)
            data_aerial[k:k + bs, :, :, 3:4] = img_util.get_data__circle(bs, circle_num=3)
        elif epoch % 8 == 5:  # np.rot90()
            k = rd.randint(ts - bs)
            data_aerial[k:k + bs] = data_aerial[k:k + bs].transpose((0, 2, 1, 3))
            data_buffer[k:k + bs] = data_buffer[k:k + bs].transpose((0, 2, 1, 3))
        else:  # shuffle mildly
            rd_j, rd_k = rd.randint(0, int(ts * 0.5 - bs), size=2)
            rd_k += int(ts * 0.5)

            data_aerial[rd_j:rd_j + bs, :, :, 0:3], data_aerial[rd_k:rd_k + bs, :, :, 0:3] = \
                data_aerial[rd_k:rd_k + bs, :, :, 0:3], data_aerial[rd_j:rd_j + bs, :, :, 0:3]
            data_buffer[rd_j:rd_j + bs], data_buffer[rd_k:rd_k + bs] = \
                data_buffer[rd_k:rd_k + bs], data_buffer[rd_j:rd_j + bs]

        for i in range(C.batch_epoch):
            j = i * bs
            feed_queue.put([data_aerial[j: j + bs],
                            data_buffer[j: j + bs], ])

            while buff_queue.qsize() > 0:
                k, buff_get = buff_queue.get()
                replace_idxs = rd.randint(0, bs, C.replace_num) if epoch != 0 else np.arange(bs)
                for replace_idx in replace_idxs:
                    data_buffer[replace_idx + k] = buff_get[replace_idx]

            # cv2.imshow('', data_aerial[0])
            # cv2.waitKey(234)
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
