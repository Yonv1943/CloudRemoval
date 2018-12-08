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
'''

C = Config('mod_defog')
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


def sess_saver_logger_old():
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


def defog_old(inp0, dim, name, reuse):
    # inp0 = tf.placeholder(tf.float32, [None, G.size, G.size, 4])  # G.size == 2 ** 8
    with tf.variable_scope(name, reuse=reuse):
        ten1 = tl.conv2d(inp0, 1 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten2 = tl.conv2d(ten1, 2 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten3 = tl.conv2d(ten2, 4 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten4 = tl.conv2d(ten3, 8 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten5 = tl.conv2d(ten4, 16 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)

        ten6 = tf.pad(ten5, paddings=tf.constant([(0, 0), (2, 2), (2, 2), (0, 0)]), mode='REFLECT')
        ten6 = tl.conv2d(ten6, 16 * dim, 3, 1, 'valid', activation=leRU_batch_norm)
        ten6 = tl.conv2d(ten6, 16 * dim, 3, 1, 'valid', activation=leRU_batch_norm)

        ten5 = tl.conv2d_transpose(tf.concat((ten6, ten5), axis=3), 32 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten4 = tl.conv2d_transpose(tf.concat((ten5, ten4), axis=3), 16 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten3 = tl.conv2d_transpose(tf.concat((ten4, ten3), axis=3), 8 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten2 = tl.conv2d_transpose(tf.concat((ten3, ten2), axis=3), 4 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten1 = tl.conv2d_transpose(tf.concat((ten2, ten1), axis=3), 2 * dim, 4, 2, 'same', activation=leRU_batch_norm)

        ten1 = tl.conv2d(tf.concat((ten1, inp0), axis=3), 1 * dim, 3, 1, 'same', activation=tf.nn.leaky_relu)
        ten1 = tl.conv2d(ten1, 4, 3, 1, 'same', activation=tf.nn.tanh)
        ten1 = ten1 * 0.505 + 0.5
        return ten1


def defog(inp0, dim, name, reuse):  # U-net
    # inp0 = tf.placeholder(tf.float32, [None, G.size, G.size, 3])  # G.size == 2 ** 8
    with tf.variable_scope(name, reuse=reuse):
        ten1 = tl.conv2d(inp0, 1 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten2 = tl.conv2d(ten1, 2 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten3 = tl.conv2d(ten2, 4 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten4 = tl.conv2d(ten3, 8 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)
        ten5 = tl.conv2d(ten4, 16 * dim, 4, 2, 'same', activation=tf.nn.leaky_relu)

        ten6 = tl.conv2d(ten5, 16 * dim, 3, 1, 'valid', activation=leRU_batch_norm)
        ten6 = tl.conv2d_transpose(ten6, 16 * dim, 3, 1, 'valid', activation=leRU_batch_norm)
        ten6 += ten5
        ten7 = tl.conv2d(ten6, 16 * dim, 3, 1, 'valid', activation=leRU_batch_norm)
        ten7 = tl.conv2d_transpose(ten7, 16 * dim, 3, 1, 'valid', activation=leRU_batch_norm)
        ten7 += ten6

        ten5 = tl.conv2d_transpose(tf.concat((ten7, ten5), axis=3), 32 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten4 = tl.conv2d_transpose(tf.concat((ten5, ten4), axis=3), 16 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten3 = tl.conv2d_transpose(tf.concat((ten4, ten3), axis=3), 8 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten2 = tl.conv2d_transpose(tf.concat((ten3, ten2), axis=3), 4 * dim, 4, 2, 'same', activation=leRU_batch_norm)
        ten1 = tl.conv2d_transpose(tf.concat((ten2, ten1), axis=3), 2 * dim, 4, 2, 'same', activation=leRU_batch_norm)

        ten1 = tl.conv2d(tf.concat((ten1, inp0), axis=3), 1 * dim, 3, 1, 'same', activation=tf.nn.leaky_relu)
        ten1 = tl.conv2d(ten1, 4, 3, 1, 'same', activation=tf.nn.tanh)
        ten1 = ten1 * 0.505 + 0.5
        return ten1


def process_train(feed_queue):  # model_train
    defog_name = 'defog'

    inp_ground = tf.placeholder(tf.float32, [None, C.size, C.size, 3])
    inp_cloud1 = tf.placeholder(tf.float32, [None, C.size, C.size, 1])
    ten_repeat = tf.ones([1, 1, 1, 3])

    inp_cloud3 = inp_cloud1 * ten_repeat
    inp_aerial = inp_ground * (1.0 - inp_cloud3) + inp_cloud3

    ten_defog4 = defog(inp_aerial, dim=8, name=defog_name, reuse=False)
    ten_cloud1 = ten_defog4[:, :, :, 0, tf.newaxis]
    ten_cloud3 = ten_cloud1 * ten_repeat
    ten_ground = ten_defog4[:, :, :, 1:4]
    ten_aerial = ten_ground * (1.0 - ten_cloud3) + ten_cloud3

    loss_cloud1 = tf.losses.absolute_difference(inp_cloud1, ten_cloud1)
    loss_aerial = tf.losses.absolute_difference(inp_aerial, ten_aerial)
    loss_defog = loss_cloud1 * 3.0 + loss_aerial

    tf_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optz_defog4 = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(loss_defog, var_list=[v for v in tf_vars if v.name.startswith(defog_name)])
    loss = [loss_cloud1, loss_aerial]
    optz = [optz_defog4, ]

    """model train"""
    sess, saver, logger, previous_train_epoch = sess_saver_logger()
    feed_dict = dict()
    eval_feed_list = feed_queue.get()
    eval_feed_dict = {inp_ground: eval_feed_list[0],
                      inp_cloud1: eval_feed_list[1], }
    eval_out_fetch = [inp_aerial,
                      inp_cloud3, ten_cloud3,
                      inp_ground, ten_ground, ]

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
                saver.save(sess, C.model_path, write_meta_graph=False)
                logger.close()
                logger = open(C.model_log, 'a')
                print(end="\n||SAVE")

                T.eval_and_get_img(mat_list=sess.run(eval_out_fetch, eval_feed_dict), channel=3,
                                   img_path=os.path.join(C.model_dir, "eval-%08d.jpg"
                                                         % (previous_train_epoch + epoch)))

    except KeyboardInterrupt:
        print("| KeyboardInterrupt")
        model_save_npy(sess, print_info=True)
    finally:
        '''save model'''
        saver.save(sess, C.model_path, write_meta_graph=False)
        print("\n| Save:", C.model_path)

        T.eval_and_get_img(mat_list=sess.run(eval_out_fetch, eval_feed_dict), channel=3,
                           img_path=os.path.join(C.model_dir, "eval-%08d.jpg"
                                                 % (previous_train_epoch + epoch)))

    print('| Batch_epoch: %dx%d' % (C.batch_epoch, C.batch_size))
    print('| Train_epoch: %d' % C.train_epoch)
    print('| TimeUsed:    %d' % int(time.time() - start_time))
    logger.close()
    sess.close()

    T.draw_plot(C.model_log)


def process_data(feed_queue):
    ts = C.train_size
    bs = C.batch_size

    data_aerial = img_util.get_data__aerial(ts, channel=3)
    data_cloud1 = img_util.get_data__cloud1(ts)
    print("||Data_sets: ready for check")
    eval_id = rd.randint(ts // 2, ts, C.eval_size * 2)
    eval_id = list(set(eval_id))[:C.eval_size]
    feed_queue.put([data_aerial[eval_id],
                    data_cloud1[eval_id]])  # for eval
    feed_queue.put([data_aerial[:bs],
                    data_cloud1[:bs], ])  # for check

    try:
        print("||Data_sets: ready for training")
        for epoch in range(C.train_epoch):
            if epoch % 8 == 0:  # np.rot90()
                data_aerial[:bs] = data_aerial[:bs].transpose((0, 2, 1, 3))
            elif epoch % 8 == 4:
                data_cloud1[:bs] = data_cloud1[:bs].transpose((0, 2, 1, 3))
            else:  # shuffle mildly
                rd_j, rd_k = rd.randint(0, int(ts * 0.5 - bs), size=2)
                rd_k += int(ts * 0.5)

                if epoch < 4:
                    data_aerial[rd_j:rd_j + bs], data_aerial[rd_k:rd_k + bs] = \
                        data_aerial[rd_k:rd_k + bs], data_aerial[rd_j:rd_j + bs]
                else:
                    data_cloud1[rd_j:rd_j + bs], data_cloud1[rd_k:rd_k + bs] = \
                        data_cloud1[rd_k:rd_k + bs], data_cloud1[rd_j:rd_j + bs]

            for i in range(C.batch_epoch):
                j = i * bs
                feed_queue.put([
                    data_aerial[j:j + bs],
                    data_cloud1[j:j + bs],
                ])

    except KeyboardInterrupt:
        print("| quit:", process_data.__name__)


def evaluation():
    cv2.namedWindow('beta', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('beta', img)
    # cv2.waitKey(1234)

    size = C.size * 2

    ground_path = '/mnt/sdb1/data_sets/AerialImageDataset/test/tyrol-e24.tif'
    ground = cv2.imread(ground_path)  # shape == (5000, 5000, 3)
    ground = ground[3000:4060, 3000:4920]

    cloud1 = img_util.get_cloud1_continusly(1943, 1943 + 1, 1)[0]
    aerial = img_util.get_aerial_continusly(ground, [cloud1, ])[0]
    repeat3 = np.ones([1, 1, 3])
    cloud3 = cloud1[:, :, np.newaxis] * repeat3
    cloud3 = cloud3.astype(np.float32) / 255.0
    cv2.imshow('beta', np.concatenate((cloud3, aerial), axis=0))
    cv2.waitKey(1234)

    tf.reset_default_graph()
    inp_aerial = tf.placeholder(tf.float32, [None, size, size, 3])
    out_defog4 = defog(inp_aerial, dim=8, name='defog', reuse=False)
    sess = sess_saver_logger()[0]

    inp = aerial[np.newaxis, -size:, -size:, :3]
    fetches = {inp_aerial: inp}
    out = sess.run(out_defog4, fetches)

    img_show0 = np.concatenate((inp[0], cloud3[-size:, -size:, :3]), axis=1)
    img_show1 = np.concatenate((out[0, :, :, 1:4], out[0, :, :, 0:1] * repeat3), axis=1)
    img_show = np.concatenate((img_show0, img_show1), axis=0)
    cv2.imshow('beta', img_show)
    cv2.waitKey(3456)

    sess.close()


def run():  # beta
    # T.draw_plot(C.model_log)
    if input("||PRESS 'y' to REMOVE model_dir? %s : " % C.model_dir) == 'y':
        shutil.rmtree(C.model_dir, ignore_errors=True)
        print("||REMOVE")
    else:
        print("||KEEP")

    import multiprocessing as mp
    feed_queue = mp.Queue(maxsize=8)

    process = [mp.Process(target=process_data, args=(feed_queue,)),
               mp.Process(target=process_train, args=(feed_queue,)), ]

    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run()
