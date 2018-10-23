import os
import time
import shutil

import cv2
import numpy as np
import numpy.random as rd
import tensorflow as tf
import tensorflow.layers as tl

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
'''

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
            print("| ymax:", y_max)

            lines.append(plt.plot(x_pts, y_pts, linestyle='dashed', marker='x', markersize=3,
                                  label='loss %d, max: %3.0f' % (idx, y_max))[0])
            plt.errorbar(x_pts, y_pts, e_pts, linestyle='None')
        plt.legend(lines, loc='upper right')
        plt.show()

    def eval_and_get_img(self, mat_list, img_path):
        mats = np.concatenate(mat_list, axis=3)
        mats = np.clip(mats, 0.0, 1.0)

        out = []
        for mat in mats:
            mat = mat.reshape((C.size, C.size, -1, 3))
            mat = mat.transpose((2, 0, 1, 3))
            mat = np.concatenate(mat, axis=0)
            mat = (mat * 255.0).astype(np.uint8)
            out.append(mat)

        img = np.concatenate(out, axis=1)
        img = np.rot90(img)
        cv2.imwrite(img_path, img)

        out = []
        for mat in mats:
            mat = mat.reshape((C.size, C.size, -1, 3))
            mat = mat.transpose((2, 0, 1, 3))


C = configure.Config('mod_GAN')
T = Tools()


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
    # inp0 = tf.placeholder(tf.float32, [None, G.size, G.size, 4])  # G.size == 2 ** 8
    with tf.variable_scope(name, reuse=reuse):
        ten1 = tl.conv2d(inp0, 1 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten2 = tl.conv2d(ten1, 2 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten3 = tl.conv2d(ten2, 4 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten4 = tl.conv2d(ten3, 8 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten5 = tl.conv2d(ten4, 16 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten6 = tl.conv2d(ten5, 32 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)

        ten7 = tf.pad(ten6, paddings=tf.constant([(0, 0), (2, 2), (2, 2), (0, 0)]), mode='REFLECT')
        ten7 = tl.conv2d(ten7, 32 * dim, 3, 1, 'valid', activation=leRU_batch_norm)
        ten7 = tl.conv2d(ten7, 32 * dim, 3, 1, 'valid', activation=leRU_batch_norm)
        ten7 = ten6 + ten7

        ten6 = tl.conv2d_transpose(ten7, 64 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten5 = tl.conv2d_transpose(ten6, 32 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten4 = tl.conv2d_transpose(ten5, 16 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten3 = tl.conv2d_transpose(ten4, 8 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten2 = tl.conv2d_transpose(ten3, 4 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten1 = tl.conv2d_transpose(ten2, 2 * dim, 4, 2, 'same', activation=leRU_batch_norm)

        ten1 = tl.conv2d(tf.concat((ten1, inp0), axis=3), 1 * dim, 3, 1, 'same', activation=leRU_batch_norm)
        ten1 = tl.conv2d(ten1, 3, 3, 1, 'same', activation=tf.nn.tanh)
        ten1 = ten1 * 0.505 + 0.5
        return ten1


def discriminator(ten, dim, name, reuse=True):
    # ans = tf.placeholder(tf.float32, [None, C.size, C.size, 3])
    with tf.variable_scope(name, reuse=reuse):
        ten = tl.conv2d(ten, dim * 1, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten = tl.conv2d(ten, dim * 2, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten = tl.conv2d(ten, dim * 4, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten = tl.conv2d(ten, dim * 8, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten = tl.conv2d(ten, dim * 16, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten = tl.conv2d(ten, dim * 32, 4, 2, 'same', activation=tf.nn.leaky_relu)

        ten = tl.conv2d(ten, dim * 64, 4, 1, 'valid', activation=tf.nn.leaky_relu)
        ten = tl.conv2d(ten, 1, 1, 1, 'valid', activation=None)
        ten = tl.flatten(ten)
        return ten


def process_train(feed_queue):  # model_train
    gene_name = 'gene'
    disc_name = 'disc'

    inp_ground = tf.placeholder(tf.float32, [None, C.size, C.size, 3])
    inp_cloud1 = tf.placeholder(tf.float32, [None, C.size, C.size, 1])
    ten_repeat = tf.ones([1, 1, 1, 3])

    ten_cloud1 = tf.clip_by_value((inp_cloud1 - 0.5) * 4.0, 0.0, 1.0)
    ten_cloud3 = ten_cloud1 * ten_repeat
    ten_cloud0 = 1.0 - ten_cloud3
    ten_aerial = inp_ground * ten_cloud0

    ten_defog4 = tf.concat([ten_aerial, inp_cloud1], axis=3)
    ten_ground = generator(ten_defog4, dim=8, name=gene_name, reuse=False)
    out_ground = ten_aerial * ten_cloud0 + ten_ground * ten_cloud3

    real_disc = discriminator(inp_ground, 2 ** 6, disc_name, reuse=False)
    fake_disc = discriminator(out_ground, 2 ** 6, disc_name, reuse=True)

    loss_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_disc, labels=tf.ones([C.batch_size, 1]), ))

    loss_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=real_disc, labels=tf.ones([C.batch_size, 1]), ))
    loss_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_disc, labels=tf.zeros([C.batch_size, 1]), ))
    loss_disc = loss_disc_real + loss_disc_fake

    tf_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optz_gene = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_gene, var_list=[v for v in tf_vars if v.name.startswith(gene_name)])
        optz_disc = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(disc_name)])
        # optz_gene = tf.train.AdamOptimizer(beta1=0.5) \
        #     .minimize(loss_gene, var_list=[v for v in tf_vars if v.name.startswith(gene_name)])
        # optz_disc = tf.train.AdamOptimizer(beta1=0.5) \
        #     .minimize(loss_disc, var_list=[v for v in tf_vars if v.name.startswith(disc_name)])

    loss = [loss_gene, loss_disc]
    optz = [optz_gene, optz_disc]

    """model train"""
    sess, saver, logger, previous_train_epoch = sess_saver_logger()
    feed_dict = dict()
    eval_out_fetch = [ten_ground, out_ground, ten_aerial]
    eval_feed_list = feed_queue.get()
    eval_feed_dict = {inp_ground: eval_feed_list[0],
                      inp_cloud1: eval_feed_list[1], }

    '''model check'''
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

            fetches = [loss, optz]

            for i in range(C.batch_epoch):
                feed_dict[inp_ground], feed_dict[inp_cloud1] = feed_queue.get()
                batch_losses.append(sess.run(fetches, feed_dict)[0])

            loss_average = np.mean(batch_losses, axis=0)
            loss_error = np.std(batch_losses, axis=0)

            logger.write('%e %e %e %e\n'
                         % (loss_average[0], loss_error[0],
                            loss_average[1], loss_error[1],))

            if time.time() - show_time > C.show_gap:
                show_time = time.time()
                remain_epoch = C.train_epoch - epoch
                remain_time = (show_time - start_time) * remain_epoch / (epoch + 1)
                print(end="\n|  %3d s |%3d epoch | Loss: %9.3e %9.3e"
                          % (remain_time, remain_epoch, loss_average[0], loss_average[1],))
            if time.time() - save_time > C.save_gap:
                '''save model'''
                save_time = time.time()
                # saver.save(sess, C.model_path, write_meta_graph=False)
                logger.close()
                logger = open(C.model_log, 'a')
                print(end="\n||SAVE")

                T.eval_and_get_img(mat_list=sess.run(eval_out_fetch, eval_feed_dict),
                                   img_path=os.path.join(C.model_dir, "eval-%08d.jpg"
                                                         % (previous_train_epoch + epoch)))
    except KeyboardInterrupt:
        print("| KeyboardInterrupt")
        saver.save(sess, C.model_path, write_meta_graph=False)
        print("| Saved to : %s" % C.model_path)
        model_save_npy(sess, print_info=False)
        print("| Saved to : %s" % C.model_npz)

    finally:
        '''save model'''
        saver.save(sess, C.model_path, write_meta_graph=False)
        print("| Saved to : %s" % C.model_path)

        T.eval_and_get_img(mat_list=sess.run(eval_out_fetch, eval_feed_dict),
                           img_path=os.path.join(C.model_dir, "eval-%08d.jpg"
                                                 % (previous_train_epoch + epoch)))

    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    print('| Train_epoch: %d' % C.train_epoch)
    print('| TimeUsed:    %d' % int(time.time() - start_time))
    logger.close()
    sess.close()

    T.draw_plot(C.model_log)


def process_data(feed_queue):
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

            if epoch % 8 == 0:
                for i, img in enumerate(aerial_data_set[:C.batch_size]):
                    aerial_data_set[i] = np.rot90(img)
            elif epoch % 8 == 4:
                for i, img in enumerate(cloud_data_set[:C.batch_size]):
                    cloud_data_set[i] = np.rot90(img)

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
        print("| quit:", process_data.__name__)


def run():  # beta
    # T.draw_plot(C.model_log)
    if input("||PRESS 'y' to REMOVE model_dir? %s : " % C.model_dir) == 'y':
        shutil.rmtree(C.model_dir, ignore_errors=True)
        print("||REMOVE")
    else:
        print("||KEEP")

    import multiprocessing as mp
    feed_queue = mp.Queue(maxsize=4)

    process = [mp.Process(target=process_data, args=(feed_queue,)),
               mp.Process(target=process_train, args=(feed_queue,)), ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run()
