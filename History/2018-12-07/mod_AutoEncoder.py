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
'''

C = Config('mod_AutoEncoderGAN')
T = img_util.Tools()
rd.seed(1943)


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
        ten = tl.conv2d(ten, 3, 3, 1, 'valid', activation=tf.nn.tanh)
        return ten * 0.501 + 0.5
        # return ten * (2**-6) - (2**-7)


def discriminator(ten, dim, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        ten = conv(ten, dim, 1)
        ten = conv(ten, dim, 2)
        ten = conv(ten, dim, 3)
        ten = tl.conv2d(ten, dim // 2, 1, 1, 'valid', activation=leru_batch_norm)
        ten = tl.conv2d(ten, dim // 4, 1, 1, 'valid', activation=leru_batch_norm)
        ten = tl.conv2d(ten, 1, 1, 1, 'valid', activation=tf.nn.tanh)
        ten = tf.reduce_mean(ten, axis=(1, 2))
        return tf.reshape(ten, (-1,))


def tf_blur(ten, blur_size=3, channel=3):
    """
    https://github.com/chiralsoftware/tensorflow/blob/master/convolve-blur.py
    """
    kernel_ary = np.zeros((blur_size, blur_size, channel, channel), np.float32)
    kernel_ary[:, :, 0, 0] = 1.0 / (blur_size ** 2)
    kernel_ary[:, :, 1, 1] = 1.0 / (blur_size ** 2)
    kernel_ary[:, :, 2, 2] = 1.0 / (blur_size ** 2)

    pad_size = blur_size // 2
    return tf.nn.conv2d(tf.pad(ten, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'REFLECT'),
                        tf.constant(kernel_ary), (1, 1, 1, 1), 'VALID')


def process_train(feed_queue):
    tf.reset_default_graph()
    name_encoder = 'encoder'
    name_decoder = 'decoder'
    name_disc = 'detail_disc'

    inp_real = tf.placeholder(tf.float32, [None, C.size, C.size, 3])
    ten_fake = encoder(inp_real, 32, name_encoder, False)
    dec_fake = decoder(ten_fake, 32, name_decoder, False)

    disc_refa = discriminator(tf.concat((inp_real, dec_fake), axis=0),
                              32, name_disc, False)

    label_real_fake = np.ones([C.batch_size * 2, ], np.float32)
    label_real_fake[C.batch_size:] = -1.0
    label_real_fake = tf.constant(label_real_fake, tf.float32)

    loss_diff = tf.losses.absolute_difference(tf_blur(inp_real), tf_blur(dec_fake))
    loss_gene = tf.losses.mean_squared_error(label_real_fake[:C.batch_size],
                                             disc_refa[C.batch_size:]) + loss_diff
    loss_disc = tf.losses.mean_squared_error(label_real_fake, disc_refa)

    tf_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optz_gene = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_gene, var_list=[v for v in tf_vars if (v.name.startswith(name_encoder),
                                                                  v.name.startswith(name_decoder),)])
        optz_disc = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(name_disc)])

        # optz = tf.train.AdamOptimizer().minimize(loss_gene, var_list=tf_vars)
        # optz_disc = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
        #     .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(disc_name)])

    loss = [loss_gene, loss_disc, loss_diff]
    optz = [optz_gene, optz_disc]

    print("||Training Initialize")
    sess, saver, logger, previous_train_epoch = mod_util.sess_saver_logger(C)
    feed_dict = dict()
    eval_out_fetch = [inp_real, dec_fake]
    eval_feed_dict = {inp_real: feed_queue.get(), }

    print("||Training Check")
    feed_list = feed_queue.get()
    feed_dict[inp_real] = feed_list
    sess.run([loss_gene, optz], feed_dict)

    print("||Training Start")
    start_time = show_time = eval_time = time.time()
    try:
        for epoch in range(C.train_epoch):
            batch_losses = list()  # init
            for i in range(C.batch_epoch):
                feed_dict[inp_real] = feed_queue.get()
                batch_losses.append(sess.run([loss, optz], feed_dict)[0])

            loss_average = np.mean(batch_losses, axis=0)
            loss_error = np.std(batch_losses, axis=0)

            logger.write('%e %e %e %e %e %e\n' % (loss_average[0], loss_error[0],
                                                  loss_average[1], loss_error[1],
                                                  loss_average[2], loss_error[2],))

            if time.time() - show_time > C.show_gap:
                show_time = time.time()
                remain_epoch = C.train_epoch - epoch
                remain_time = (show_time - start_time) * remain_epoch / (epoch + 1)
                print(end="\n|  %3d s |%3d epoch | Loss: %9.3e %9.3e %9.3e"
                          % (remain_time, remain_epoch,
                             loss_average[0], loss_average[1], loss_average[2],))
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

    print('| Train_epoch: %d' % C.train_epoch)
    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    print('| TimeUsed:    %d' % int(time.time() - start_time))
    logger.close()
    sess.close()

    T.draw_plot(C.model_log)


def process_data(feed_queue):
    ts = C.train_size
    bs = C.batch_size

    timer = time.time()
    aerials = img_util.get_data__aerial(ts, channel=3)

    print("||Data_sets: ready for check |Used time:", int(time.time() - timer))
    eval_id = rd.randint(ts // 2, ts, C.eval_size * 2)
    eval_id = list(set(eval_id))[:C.eval_size]
    feed_queue.put(aerials[eval_id])  # for eval
    feed_queue.put(aerials[:bs])  # for check

    print("||Data_sets: ready for training")
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

        for i in range(C.batch_epoch):
            j = i * bs
            feed_queue.put(aerials[j: j + bs])
            # cv2.imshow('', aerials[0])
            # cv2.waitKey(234)
    print("| quit:", process_data.__name__)


def run():  # beta
    # T.draw_plot(C.model_log)
    print('| Train_epoch: %d' % C.train_epoch)
    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
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

    process = [mp.Process(target=process_data, args=(feed_queue,)),
               mp.Process(target=process_train, args=(feed_queue,)), ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run()
