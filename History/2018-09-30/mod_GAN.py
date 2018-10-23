import os
import time
import shutil

import cv2
import numpy as np
import numpy.random as rd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore the TensorFlow log messages
import tensorflow as tf
import configure
from util.img_util import get_data_sets


class Tools(object):
    def ten_check(self, ten):
        return tf.Print(ten, [str(ten.shape)], message='|| ')

    def img_check(self, img):
        print("| min,max %6.2f %6.2f |%s", img.shape, np.min(img), np.max(img))

    def ary_check(self, ary):
        print("| min,max %6.2f %6.2f |ave,std %6.2f %6.2f |%s" %
              (np.min(ary), np.max(ary), np.average(ary), float(np.std(ary)), ary.shape,))

    def draw_plot(self, log_txt_path):
        print("||" + self.draw_plot.__name__)
        if not os.path.exists(log_txt_path):  # check
            print("|NotExist:", log_txt_path)
            return None

        arys = np.loadtxt(log_txt_path)
        if arys.shape[0] < 2:
            print("|Empty:", log_txt_path)
            return None

        if 'plt' not in globals():
            import matplotlib.pyplot as plt_global
            global plt
            plt = plt_global

        arys_len = int(len(arys) * 0.9)
        arys = arys[-arys_len:]
        arys = arys.reshape((arys_len, -1, 2)).transpose((1, 0, 2))

        lines = []
        for idx, ary in enumerate(arys):
            x_pts = [i for i in range(ary.shape[0])]
            y_pts = ary[:, 0]
            e_pts = ary[:, 1]

            y_max = y_pts.max() + 2 ** -32
            y_pts /= y_max
            e_pts /= y_max

            lines.append(plt.plot(x_pts, y_pts, linestyle='dashed', marker='x', markersize=3,
                                  label='loss %d, max: %3.0f' % (idx, y_max))[0])
            plt.errorbar(x_pts, y_pts, e_pts, linestyle='None')
        plt.legend(lines, loc='upper right')
        plt.show()


C = configure.Config()
T = Tools()


def mat2img(mats):
    mats = np.clip(mats, 0.0, 1.0)
    out = []
    for mat in mats:
        aerial = mat[:, :, 0:3]
        cloud1I = mat[:, :, 3, np.newaxis].repeat(3, axis=2)
        cloud1T = mat[:, :, 4, np.newaxis].repeat(3, axis=2)
        groundI = mat[:, :, 5:8]
        groundT = mat[:, :, 8:11]
        groundO = mat[:, :, 11:14]

        img_show = (aerial, cloud1I, cloud1T, groundI, groundT, groundO)
        img_show = np.concatenate(img_show, axis=0)
        img_show = (img_show * 255.0).astype(np.uint8)
        out.append(img_show)
    res = np.concatenate(out, axis=1)
    return res


def sess_saver_logger():
    # config = tf.ConfigProto()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=C.gpu_limit))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=4)
    if os.path.exists(os.path.join(C.model_dir, 'checkpoint')):
        C.model_path = tf.train.latest_checkpoint(C.model_dir)
        saver.restore(sess, C.model_path)
        print("| Load from checkpoint:", C.model_path)
    elif os.path.exists(C.model_npz):
        # sess.run(tf.global_variables_initializer())
        # sess.run([i.assign(np.load(G.model_npz)[i.name]) for i in tf.trainable_variables()])
        sess.run([i.assign(np.load(C.model_npz)[i.name]) for i in tf.global_variables()])
        print("| Load from npz:", C.model_path)
    else:
        os.makedirs(C.model_dir, exist_ok=True)
        sess.run(tf.global_variables_initializer())
        print("| Init:", C.model_path)

    logger = open(C.model_log, 'a')
    previous_train_epoch = sum(1 for _ in open(C.model_log)) if os.path.exists(C.model_log) else 0
    print('| Train_epoch: %6d+%6d' % (previous_train_epoch, C.train_epoch))
    print('| Batch_epoch: %6dx%6d' % (C.batch_epoch, C.batch_size))
    return sess, saver, logger, previous_train_epoch


def conv2d(ten, filters):
    ten = tf.layers.conv2d(ten, filters, kernel_size=4, strides=2,
                           padding='same', activation=tf.nn.leaky_relu)
    return ten


def conv2d_tp(ten, filters):
    ten = tf.layers.conv2d_transpose(ten, filters, kernel_size=4, strides=2,
                                     padding='same', activation=tf.nn.leaky_relu)
    return ten


def defog(inp0, filters, out_dim, name_scope, reuse):
    # inp0 = tf.placeholder(tf.float32, [None, G.size, G.size, 4])  # G.size == 2 ** 8
    with tf.variable_scope(name_scope, reuse=reuse):
        # inp0 = tf.contrib.layers.batch_norm(inp0)
        ten1 = conv2d(inp0, filters * 1)
        ten2 = conv2d(ten1, filters * 2)
        ten3 = conv2d(ten2, filters * 4)
        ten4 = conv2d(ten3, filters * 8)
        ten5 = conv2d(ten4, filters * 16)

        ten5 = tf.contrib.layers.batch_norm(ten5)

        ten5 = conv2d_tp(ten5, filters * 8)
        ten4 = conv2d_tp(tf.concat((ten5, ten4), axis=3), filters * 16)
        ten3 = conv2d_tp(tf.concat((ten4, ten3), axis=3), filters * 8)
        ten2 = conv2d_tp(tf.concat((ten3, ten2), axis=3), filters * 4)
        ten1 = conv2d_tp(tf.concat((ten2, ten1), axis=3), filters * 2)

        out = tf.concat((inp0, ten1), axis=3)
        out = tf.layers.conv2d(out, filters, 3, 1, padding='same', activation=tf.nn.leaky_relu)
        out = tf.layers.conv2d(out, out_dim, 1, 1, padding='same', activation=tf.nn.tanh)
        out = out * 0.505 + 0.5
    return out


def discriminator(inp0, filters, name_scope, reuse):
    with tf.variable_scope(name_scope, reuse=reuse):
        # inp0 = tf.contrib.layers.batch_norm(inp0)
        ten = conv2d(inp0, filters * 1)
        ten = conv2d(ten, filters * 2)
        ten = conv2d(ten, filters * 4)
        ten = conv2d(ten, filters * 8)
        ten = conv2d(ten, filters * 16)

        out = tf.layers.conv2d(ten, 1, 1, padding='valid')
        out = tf.reshape(out, (-1, ))
    return out


def get_feed_queue(feed_queue):  # model_train
    """model init"""
    '''defog'''
    inp_ground = tf.placeholder(tf.float32, [None, C.size, C.size, 3])
    inp_cloud1 = tf.placeholder(tf.float32, [None, C.size, C.size, 1])
    ten_repeat = tf.ones([1, 1, 1, 3])

    inp_cloud3 = inp_cloud1 * ten_repeat
    inp_aerial = inp_ground * (1.0 - inp_cloud3) + inp_cloud3

    ten_cldgrd = defog(inp_aerial, filters=8, out_dim=4, name_scope='defog_cloud', reuse=False)
    ten_cloud1 = ten_cldgrd[:, :, :, 0, tf.newaxis]
    ten_cloud3 = ten_cloud1 * ten_repeat
    ten_ground = ten_cldgrd[:, :, :, 1:4]
    ten_aerial = ten_ground * (1.0 - ten_cloud3) + ten_cloud3

    loss_cloud1 = tf.losses.absolute_difference(inp_cloud1, ten_cloud1)
    loss_aerial = tf.losses.absolute_difference(inp_aerial, ten_aerial)
    loss_defog = loss_cloud1 + loss_aerial

    '''mend GAN'''
    out_ground = defog(ten_cldgrd, filters=12, out_dim=3, name_scope='mend_gene', reuse=False)
    # out_ground = ten_ground * (1.0 - ten_cloud3) + out_ground * ten_cloud3

    ans_real = tf.ones(C.batch_size, tf.float32)
    ans_fake = tf.zeros(C.batch_size, tf.float32)

    disc_real = discriminator(inp_ground, 8, name_scope='mend_disc', reuse=False)
    disc_fake = discriminator(out_ground, 8, name_scope='mend_disc', reuse=True)

    loss_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ans_real, logits=disc_real))
    loss_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ans_fake, logits=disc_fake))
    loss_disc = loss_disc_real + loss_disc_fake

    loss_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=ans_real, logits=disc_fake))

    '''optimizer'''
    tf_vars = tf.trainable_variables()
    var_defog = [v for v in tf_vars if v.name.startswith('defog_cloud')]
    var_gene = [v for v in tf_vars if v.name.startswith('mend_gene')]
    var_disc = [v for v in tf_vars if v.name.startswith('mend_disc')]

    optz_conv = tf.train.AdamOptimizer(C.learning_rate) \
        .minimize(loss_defog, var_list=var_defog)
    optz_gene = tf.train.AdamOptimizer(C.learning_rate) \
        .minimize(loss_gene, var_list=var_gene)
    optz_disc = tf.train.AdamOptimizer(C.learning_rate) \
        .minimize(loss_disc, var_list=var_disc)

    loss = [loss_defog, loss_gene, loss_disc]
    optz = [optz_conv, optz_gene, optz_disc]

    """model train"""
    sess, saver, logger, previous_train_epoch = sess_saver_logger()
    feed_dict = dict()
    eval_feed_list = feed_queue.get()
    eval_feed_dict = {inp_ground: eval_feed_list[0],
                      inp_cloud1: eval_feed_list[1], }

    '''model check'''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # recover the TensorFlow log messages
    print("||Training Check")
    feed_list = feed_queue.get()
    feed_dict[inp_ground] = feed_list[0]
    feed_dict[inp_cloud1] = feed_list[1]
    sess.run([loss, optz], feed_dict)

    epoch = 0
    start_time = show_time = save_time = time.time()
    try:
        print("||Training Start")
        for epoch in range(C.train_epoch):
            batch_losses = list()

            dim4_loss, disc_loss = sess.run([loss_defog, loss_disc], feed_dict)
            pass_mend = bool(dim4_loss > C.max_gene_loss)
            pass_disc = bool(disc_loss < C.min_disc_loss)

            fetches = [loss, optz]
            if pass_mend:
                fetches = [loss[:1], optz[:1]]
            elif pass_disc:
                fetches = [loss[:2], optz[:2]]

            for i in range(C.batch_epoch):
                feed_dict[inp_ground], feed_dict[inp_cloud1] = feed_queue.get()
                batch_losses.append(sess.run(fetches, feed_dict)[0])

            loss_average = np.mean(batch_losses, axis=0)
            loss_error = np.std(batch_losses, axis=0)
            if pass_mend:
                loss_average = np.concatenate((loss_average, [0.0, disc_loss]), axis=0)
                loss_error = np.concatenate((loss_error, [0.0, 0.0]), axis=0)
            elif pass_disc:
                loss_average = np.concatenate((loss_average, [disc_loss, ]), axis=0)
                loss_error = np.concatenate((loss_error, [0.0, ]), axis=0)

            logger.write('%e %e %e %e %e %e\n'
                         % (loss_average[0], loss_error[0],
                            loss_average[1], loss_error[1],
                            loss_average[2], loss_error[2],))

            if time.time() - show_time > C.show_gap:
                show_time = time.time()
                remain_epoch = C.train_epoch - epoch
                remain_time = (show_time - start_time) * remain_epoch / (epoch + 1)
                print(end="\n|  %3d s |%3d epoch | Loss: %9.3e %9.3e %9.3e |%d %d"
                          % (remain_time, remain_epoch,
                             loss_average[0], loss_average[1], loss_average[2],
                             int(pass_mend), int(pass_disc)))
            if time.time() - save_time > C.save_gap:
                '''save model'''
                save_time = time.time()
                saver.save(sess, C.model_path, write_meta_graph=False)
                logger.close()
                logger = open(C.model_log, 'a')
                print(end="\n||SAVE")

                '''eval while training'''
                eval_out = sess.run([inp_aerial, inp_cloud1, ten_cloud1,
                                     inp_ground, ten_ground, out_ground], eval_feed_dict)

                img_show = np.concatenate(eval_out, axis=3)
                img_show = mat2img(img_show)
                cv2.imwrite(os.path.join(C.model_dir, "eval-%08d.jpg" % (previous_train_epoch + epoch)), img_show)
    except KeyboardInterrupt:
        print("| KeyboardInterrupt")
    finally:
        '''save model'''
        saver.save(sess, C.model_path, write_meta_graph=False)
        print("\n| Save:", C.model_path)

        '''eval while training'''
        eval_out = sess.run([inp_aerial, inp_cloud1, ten_cloud1,
                             inp_ground, ten_ground, out_ground], eval_feed_dict)

        img_show = np.concatenate(eval_out, axis=3)
        img_show = mat2img(img_show)
        cv2.imwrite(os.path.join(C.model_dir, "eval-%08d.jpg" % (previous_train_epoch + epoch)), img_show)

    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    print('| Train_epoch: %d' % C.train_epoch)
    print('| TimeUsed:    %d' % int(time.time() - start_time))
    logger.close()
    sess.close()

    T.draw_plot(C.model_log)


def put_feed_queue(feed_queue):
    aerial_data_set, cloud_data_set = get_data_sets(C.train_size)
    print("||Data_sets: ready for check")

    rd.shuffle(aerial_data_set)
    feed_queue.put([aerial_data_set[:C.test_size],
                    cloud_data_set[:C.test_size], ])  # for eval
    feed_queue.put([aerial_data_set[0: 0 + C.batch_size],
                    cloud_data_set[0: 0 + C.batch_size], ])  # for check
    try:
        print("||Data_sets: ready for training")
        for epoch in range(C.train_epoch):
            if epoch % 2 == 0:
                rd.shuffle(aerial_data_set)
            else:
                rd.shuffle(cloud_data_set)

            for i in range(C.batch_epoch):
                j = i * C.batch_size
                feed_list = [aerial_data_set[j: j + C.batch_size],
                             cloud_data_set[j: j + C.batch_size], ]

                feed_queue.put(feed_list)

                # img_ground = feed_list[0][:G.test_size]
                # img_cloudI = feed_list[1][:G.test_size]
                # mats = np.concatenate((img_ground, img_cloudI, img_ground), axis=3)
                # cv2.imshow('', mat2img(mats))
                # cv2.waitKey(234)

    except KeyboardInterrupt:
        print("| quit:", put_feed_queue.__name__)


def run():  # beta
    C.model_dir = 'mod_GAN'
    C.model_name = 'mod'
    C.model_path = os.path.join(C.model_dir, C.model_name)
    C.model_npz = os.path.join(C.model_dir, C.model_name + '.npz')
    C.model_log = os.path.join(C.model_dir, 'training_npy.txt')

    if input("||REMOVE model_dir? %s\n||PRESS 'y' to REMOVE: " % C.model_dir) == 'y':
        shutil.rmtree(C.model_dir, ignore_errors=True)
        print("||REMOVE")
    else:
        print("||KEEP")

    import multiprocessing as mp
    feed_queue = mp.Queue(maxsize=4)

    process = [mp.Process(target=put_feed_queue, args=(feed_queue,)),
               mp.Process(target=get_feed_queue, args=(feed_queue,)), ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run()
