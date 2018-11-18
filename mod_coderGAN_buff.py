import os
import time
import shutil

import numpy as np
import numpy.random as rd
import tensorflow as tf
import tensorflow.layers as tl

from configure import Config
from util import img_util
from util import mod_util

'''
2018-10-10  Yonv1943

2018-11-14  kernel2 better than kernel4, but low solution
2018-11-15  add conv_tp_conv to decoder
2018-11-15  Debug: np.savez(C.model_npz, **npy_dict) FOR mod_save_npy()
2018-11-15  Debug: load from mod.npz FOR mod_sess_saver_logger()
'''

C = Config('mod_coderGAN_buff')
T = img_util.Tools()


def leru_batch_norm(ten):
    ten = tf.layers.batch_normalization(ten, training=True)
    ten = tf.nn.leaky_relu(ten)
    return ten


def conv(ten, dim, idx):
    filters = (2 ** (idx - 1)) * dim
    return tl.conv2d(ten, filters, 3, 2, 'same', activation=tf.nn.leaky_relu)


def conv_tp(ten, dim, idx):
    filters = (2 ** idx) * dim
    return tl.conv2d_transpose(ten, filters, 3, 2, 'same', activation=leru_batch_norm)


def conv_tp_conv(ten, filters):
    ten = tl.conv2d_transpose(ten, filters, 3, 1, 'valid', activation=leru_batch_norm)
    ten = tl.conv2d(ten, filters, 3, 1, 'valid', activation=tf.nn.leaky_relu)
    return ten


def encoder(ten, dim, name, reuse):
    # inp = tf.placeholder(tf.float32, [None, G.size, G.size, 3])  # G.size > 2 ** 5 + 1
    with tf.variable_scope(name, reuse=reuse):
        ten = conv(ten, dim, 1)
        ten = conv(ten, dim, 2)
        ten = conv(ten, dim, 3)
        ten = conv(ten, dim, 4)
        ten = conv(ten, dim, 5)

        filters = (2 ** (5 - 1)) * dim
        out = tl.conv2d_transpose(ten, filters, 3, 1, 'valid', activation=leru_batch_norm)
        out = tl.conv2d(out, filters, 3, 1, 'valid', activation=tf.nn.leaky_relu)
        return out + ten


def decoder(ten, dim, name, reuse):
    # out = tf.placeholder(tf.float32, [None, G.size, G.size, 3])  # G.size > 2 ** 5 + 1
    with tf.variable_scope(name, reuse=reuse):
        ten = conv_tp(ten, dim, 5)
        ten = conv_tp(ten, dim, 4)
        ten = conv_tp(ten, dim, 3)
        ten = conv_tp(ten, dim, 2)
        ten = conv_tp(ten, dim, 1)

        ten = conv_tp_conv(ten, dim * 2) + ten
        ten = tl.conv2d_transpose(ten, dim, 3, 1, 'valid', activation=leru_batch_norm)
        ten = tl.conv2d(ten, 3, 3, 1, 'valid', activation=tf.nn.sigmoid)
        return ten * 1.01 - 0.005
        # return ten * (2**-6) - (2**-7)


def discriminator(ten, dim, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        ten = conv_tp_conv(ten, dim)
        ten = conv_tp_conv(ten, dim)
        ten = conv_tp_conv(ten, dim)
        ten = conv_tp_conv(ten, dim)
        ten = tl.conv2d(ten, 1, 3, 1, 'valid')
        ten = tf.reduce_mean(ten, axis=(1, 2))
        ten = tf.nn.tanh(ten)
        return tf.reshape(ten, (-1,))


def generator(ten, dim, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        ten = conv_tp_conv(ten, dim + 1) + ten
        ten = conv_tp_conv(ten, dim + 1) + ten
        ten = conv_tp_conv(ten, dim + 1) + ten
        ten = conv_tp_conv(ten, dim + 1) + ten
        ten = conv_tp_conv(ten, dim + 1) + ten
        ten = conv_tp_conv(ten, dim + 1) + ten
    return conv_tp_conv(ten, dim)[:, :, :, :dim]


def process_train(feed_queue, buff_queue0, buff_queue1):  # model_train
    tf.reset_default_graph()
    name_encoder = 'encoder'
    name_decoder = 'decoder'
    name_gene = 'gene'
    name_disc = 'disc'

    inp_aerial = tf.placeholder(tf.float32, [None, C.size, C.size, 3])
    inp_mask01 = tf.placeholder(tf.float32, [None, C.size, C.size, 1])

    ten_repeat = tf.ones([1, 1, 1, 3], tf.float32)
    ten_mask03 = inp_mask01 * ten_repeat
    ten_ragged = inp_aerial * (1.0 - ten_mask03)

    ten_ragg = encoder(ten_ragged, 32, name_encoder, False)
    ten_real = encoder(inp_aerial, 32, name_encoder, True)

    size, dim = ten_ragg.get_shape().as_list()[2:4]
    print("||Neck, size dim:", size, dim)
    inp_buff = tf.placeholder_with_default(np.zeros((C.batch_size, 8, 8, 512), np.float32),
                                           [None, size, size, dim])
    ten_mask = tf.image.resize_images(inp_mask01, (size, size))

    ten_fake = generator(tf.concat((ten_ragg, ten_mask), axis=3),
                         dim, name_gene, False)

    disc_refa = discriminator(tf.concat((ten_real, ten_fake), axis=0),
                              dim, name_disc, False)
    disc_rebu = discriminator(tf.concat((ten_real, inp_buff), axis=0),
                              dim, name_disc, True)

    dec_fake = decoder(ten_fake, 32, name_decoder, False)
    dec_real = decoder(ten_real, 32, name_decoder, True)
    dec_ragg = decoder(ten_ragg, 32, name_decoder, True)

    loss_diff = tf.losses.absolute_difference((dec_ragg - dec_fake) * (1.0 - ten_mask03),
                                              tf.zeros_like(dec_fake))

    '''loss'''
    label_real_fake = np.ones([C.batch_size * 2, ], np.float32)
    label_real_fake[C.batch_size:] = -1.0
    label_real_fake = tf.constant(label_real_fake, tf.float32)

    loss_gene = tf.losses.mean_squared_error(label_real_fake[:C.batch_size],
                                             disc_refa[C.batch_size:])
    # loss_gene = tf.losses.mean_squared_error(label_real_fake[:C.batch_size],
    #                                          disc_refa[C.batch_size:]) + loss_diff

    loss_disc = tf.losses.mean_squared_error(label_real_fake, disc_refa) + \
                tf.losses.mean_squared_error(label_real_fake, disc_rebu)

    # '''loss'''
    # label_ones = tf.ones([C.batch_size, ], tf.float32)
    # label_zero = tf.zeros([C.batch_size, ], tf.float32)
    #
    # loss_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     logits=disc_fake, labels=label_ones, ))
    #
    # loss_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     logits=disc_real, labels=label_ones, ))
    # loss_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     logits=disc_fake, labels=label_zero, ))
    # loss_disc = loss_disc_real + loss_disc_fake

    # '''buff'''
    # disc_buff = discriminator(inp_buff, rep_mask01, 32, name_disc, True)
    # loss_buff = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     logits=disc_buff, labels=label_zero, ))

    '''optz'''
    tf_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optz_gene = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_gene, var_list=[v for v in tf_vars if v.name.startswith(name_gene)])
        optz_disc = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(name_disc)])
        # optz_buff = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
        #     .minimize(loss_buff, var_list=[v for v in tf_vars if v.name.startswith(name_disc)])

    loss = [loss_gene, loss_disc]
    optz = [optz_gene, optz_disc]

    print("||Training Initialize")
    sess, saver, logger, previous_train_epoch = mod_util.sess_saver_logger(C)
    eval_out_fetch = [dec_real, dec_fake, dec_ragg]
    eval_feed_list = feed_queue.get()
    eval_feed_dict = {inp_aerial: eval_feed_list[0],
                      inp_mask01: eval_feed_list[1], }

    print("||Training Check")
    feed_dict = dict()
    feed_list = feed_queue.get()
    feed_dict[inp_aerial], feed_dict[inp_mask01], = feed_list
    ary_buff, batch_loss = sess.run([ten_fake, loss, optz], feed_dict)[:2]
    buff_queue1.put(ary_buff)

    print("||Training Start")
    start_time = show_time = eval_time = time.time()
    try:
        for epoch in range(C.train_epoch):
            batch_losses = list()  # init
            for i in range(C.batch_epoch):
                feed_dict[inp_aerial], feed_dict[inp_mask01] = feed_queue.get()
                feed_dict[inp_buff] = buff_queue0.get()

                # print(bool(batch_loss[0] > batch_loss[1] * C.active_rate), batch_loss)
                if batch_loss[0] > batch_loss[1] * C.active_rate:
                    optz = optz_gene
                else:
                    optz = [optz_gene, optz_disc]

                ary_buff, batch_loss = sess.run([ten_fake, loss, optz], feed_dict)[:2]

                buff_queue1.put(ary_buff)
                batch_losses.append(batch_loss)

            loss_average = np.mean(batch_losses, axis=0)
            loss_error = np.std(batch_losses, axis=0)

            logger.write('%e %e %e %e\n' % (loss_average[0], loss_error[0],
                                            loss_average[1], loss_error[1],))

            if time.time() - show_time > C.show_gap:
                show_time = time.time()
                remain_epoch = C.train_epoch - epoch
                remain_time = (show_time - start_time) * remain_epoch / (epoch + 1)
                print(end="\n|  %3d s |%3d epoch | Loss: %9.3e %9.3e"
                          % (remain_time, remain_epoch,
                             loss_average[0], loss_average[1],))
            if time.time() - eval_time > C.eval_gap:
                eval_time = time.time()
                logger.close()
                logger = open(C.model_log, 'a')
                print(end="\n||EVAL")

                T.eval_and_get_img(mat_list=sess.run(eval_out_fetch, eval_feed_dict), channel=3,
                                   img_path=os.path.join(C.model_dir, "eval-%08d.jpg"
                                                         % (previous_train_epoch + epoch)), )
    except KeyboardInterrupt:
        print("| KeyboardInterrupt")

    saver.save(sess, C.model_path, write_meta_graph=False)
    print("| SAVE: %s" % C.model_path)
    mod_util.save_npy(sess, C.model_npz)
    T.eval_and_get_img(mat_list=sess.run(eval_out_fetch, eval_feed_dict), channel=3,
                       img_path=os.path.join(C.model_dir, "eval.jpg"))

    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    print('| Train_epoch: %d' % C.train_epoch)
    print('| TimeUsed:    %d' % int(time.time() - start_time))
    logger.close()
    sess.close()

    T.draw_plot(C.model_log)


def process_data(feed_queue):
    ts = C.train_size
    bs = C.batch_size

    timer = time.time()
    aerials = img_util.get_data__aerial(ts, channel=3)
    mask01s = img_util.get_data__circle(ts, circle_num=3)

    print("| Data_sets: ready for check |Used time:", int(time.time() - timer))
    eval_id = rd.randint(ts // 2, ts, C.eval_size * 2)
    eval_id = list(set(eval_id))[:C.eval_size]
    feed_queue.put([aerials[eval_id],
                    mask01s[eval_id], ])  # for eval
    feed_queue.put([aerials[:bs],
                    mask01s[:bs], ])  # for check

    print("| Data_sets: ready for training")
    for epoch in range(C.train_epoch):
        # if epoch % 8 == 1:  # refresh circle(mask)
        #     k = rd.randint(ts - bs)
        #     aerials[k:k + bs, :, :, 3:4] = img_util.get_data__circle(bs, circle_num=3)
        # elif epoch % 8 == 5:  # np.rot90()
        #     k = rd.randint(ts - bs)
        #     aerials[k:k + bs] = aerials[k:k + bs].transpose((0, 2, 1, 3))
        #     data_buffer[k:k + bs] = data_buffer[k:k + bs].transpose((0, 2, 1, 3))
        #     data_mask01[k:k + bs] = data_mask01[k:k + bs].transpose((0, 2, 1, 3))
        # else:  # shuffle mildly
        rd_j, rd_k = rd.randint(0, int(ts * 0.5 - bs), size=2)
        rd_k += int(ts * 0.5)

        aerials[rd_j:rd_j + bs], aerials[rd_k:rd_k + bs] = \
            aerials[rd_k:rd_k + bs], aerials[rd_j:rd_j + bs]
        mask01s[rd_j:rd_j + bs], mask01s[rd_k:rd_k + bs] = \
            mask01s[rd_k:rd_k + bs], mask01s[rd_j:rd_j + bs]

        for i in range(C.batch_epoch):
            j = i * bs
            feed_queue.put([aerials[j: j + bs],
                            mask01s[j: j + bs], ])
            # cv2.imshow('', aerials[0])
            # cv2.waitKey(234)
    print("| quit:", process_data.__name__)


def process_buff(buff_queue0, buff_queue1):
    ts = C.train_size
    bs = C.batch_size

    size, dim = buff_queue1.get().shape[2:4]
    buffers = np.zeros((ts, size, size, dim), np.float32)

    for epoch in range(C.train_epoch):
        for i in range(C.batch_epoch):
            j = i * bs
            buff_queue0.put(buffers[j: j + bs])

            buff_get = buff_queue1.get()
            replace_ids = rd.randint(0, bs, C.replace_num) if epoch != 0 else np.arange(bs)
            for replace_idx in replace_ids:
                buffers[replace_idx + j] = buff_get[replace_idx]


def run():  # beta
    # T.draw_plot(C.model_log)
    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    print('| Train_epoch: %d' % C.train_epoch)
    if input("||PRESS 'y' to REMOVE model_dir? %s: " % C.model_dir) == 'y':
        shutil.rmtree(C.model_dir, ignore_errors=True)
        print("||Remove")
    elif input("||PRESS 'y' to UPDATE model_npz? %s: " % C.model_npz) == 'y':
        mod_util.update_npz(src_path='mod_AutoEncoder/mod.npz', dst_path=C.model_npz)

        remove_path = os.path.join(C.model_dir, 'checkpoint')
        os.remove(remove_path) if os.path.exists(remove_path) else None
        # shutil.rmtree(os.path.join(C.model_dir, 'checkpoint'), ignore_errors=True)

    import multiprocessing as mp
    feed_queue = mp.Queue(maxsize=8)
    buff_queue0 = mp.Queue(maxsize=8)
    buff_queue1 = mp.Queue(maxsize=8)

    process = [mp.Process(target=process_data, args=(feed_queue,)),
               mp.Process(target=process_buff, args=(buff_queue0, buff_queue1)),
               mp.Process(target=process_train, args=(feed_queue, buff_queue0, buff_queue1)), ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run()
