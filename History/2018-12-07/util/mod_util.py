import os
import numpy as np
import tensorflow as tf

from configure import Config

'''
2018-10-10  Yonv1943
2018-11-15  def update_npz()
2018-11-16  tf_config
'''

np.random.seed(1943)


def save_npy(sess, model_npz):
    tf_vars = tf.global_variables()

    '''save as singal npy'''
    npy_dict = dict()
    for var in tf_vars:
        npy_dict[var.name] = var.eval(session=sess)
        # print("| FETCH: %s" % var.name) if print_info else None
    np.savez(model_npz, **npy_dict)
    with open(model_npz + '.txt', 'w') as f:
        f.writelines(["%s\n" % key for key in npy_dict.keys()])

    # '''save as several npy'''
    # shutil.rmtree(C.model_npy, ignore_errors=True)
    # os.makedirs(C.model_npy, exist_ok=True)
    # for v in tf_vars:
    #     v_name = str(v.name).replace('/', '-').replace(':', '.') + '.npy'
    #     np.save(os.path.join(C.model_npy, v_name), v.eval(session=sess))
    #     print("| SAVE %s.npy" % v.name) if print_info else None
    print("| SAVE: %s" % model_npz)


def update_npz(src_path, dst_path):
    # src_path = 'mod_AutoEncoder/mod.npz'

    src_npz = np.load(src_path)
    src_key = set(src_npz.keys())

    dst_ary = dict(np.load(dst_path))
    dst_key = set(dst_ary.keys())

    for key in dst_key & src_key:
        if src_npz[key].shape == dst_ary[key].shape:
            dst_ary[key] = src_npz[key]
            print("  update:", key)
    np.savez(dst_path, **dst_ary)
    print("| Update:", dst_path)


def load_npy(sess, model_npz):  # Not Enabled
    tf_dict = dict()
    for tf_var in tf.global_variables():
        tf_dict[tf_var.name] = tf_var

    for npy_name in os.listdir(model_npz):
        var_name = npy_name[:-4].replace('-', '/').replace('.', ':')
        var_node = tf_dict.get(var_name, None)
        if var_node:
            var_ary = np.load(os.path.join(model_npz, npy_name))
            sess.run(tf.assign(var_node, var_ary)) if var_node else None

    if os.path.exists(model_npz):
        sess.run(tf.global_variables_initializer())
        # sess.run([i.assign(np.load(C.model_npz)[i.name]) for i in tf.trainable_variables()])
        sess.run([i.assign(np.load(model_npz)[i.name]) for i in tf.global_variables()])
        print("| Load from npz:", model_npz)


def sess_saver_logger(c):
    tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=c.gpu_limit))
    tf_config.gpu_options.allow_growth = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(c.gpu_id)  # choose GPU:0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore the TensorFlow log messages
    sess = tf.Session(config=tf_config)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # recover the TensorFlow log messages

    saver = tf.train.Saver(max_to_keep=4)
    if os.path.exists(os.path.join(c.model_dir, 'checkpoint')):
        # C.model_path = tf.train.latest_checkpoint(C.model_dir)
        saver.restore(sess, c.model_path)
        print("||Load from checkpoint:", c.model_path)
    elif os.path.exists(c.model_npz):
        name2ary = np.load(c.model_npz)
        print("||Load from mod.npz")
        sess.run(tf.global_variables_initializer())
        for var_node in tf.global_variables():
            var_ary = name2ary[var_node.name]
            sess.run(tf.assign(var_node, var_ary))
    else:  # Initialize
        os.makedirs(c.model_dir, exist_ok=True)
        sess.run(tf.global_variables_initializer())
        print("||Init:", c.model_path)

    logger = open(c.model_log, 'a')
    previous_train_epoch = sum(1 for _ in open(c.model_log)) if os.path.exists(c.model_log) else 0
    print('| Train_epoch: %6d+%6d' % (previous_train_epoch, c.train_epoch))
    print('| Batch_epoch: %6dx%6d' % (c.batch_epoch, c.batch_size))
    return sess, saver, logger, previous_train_epoch


def tf_test():
    """
    https://github.com/chiralsoftware/tensorflow/blob/master/convolve-blur.py
    """
    import cv2
    from util.img_util import Tools
    from configure import Config

    C = Config()
    T = Tools()

    cv2.namedWindow('beta', cv2.WINDOW_KEEPRATIO)
    # img = cv2.imread(os.path.join(C.aerial_dir, 'bellingham1.tif'))  # test
    img = cv2.imread(os.path.join(C.aerial_dir, 'austin1.tif'))  # train
    img = img[np.newaxis, :C.size, :C.size, :3]

    blur_size = 3
    channel = 3
    kernel_ary = np.zeros((blur_size, blur_size, channel, channel), np.float32)
    kernel_ary[:, :, 0, 0] = 1.0 / (blur_size ** 2)
    kernel_ary[:, :, 1, 1] = 1.0 / (blur_size ** 2)
    kernel_ary[:, :, 2, 2] = 1.0 / (blur_size ** 2)

    inp = tf.placeholder(tf.float32, [None, C.size, C.size, 3])
    ten = tf.nn.conv2d(tf.pad(inp, ((0, 0), (2, 2), (2, 2), (0, 0)), 'REFLECT'),
                       tf.constant(kernel_ary), (1, 1, 1, 1), 'VALID')

    tf_config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=tf_config)
    img = sess.run(ten, {inp: img})
    sess.close()

    img = np.array(img[0], np.uint8)
    T.img_check(img)
    cv2.imshow('beta', img)
    cv2.waitKey(3456)
