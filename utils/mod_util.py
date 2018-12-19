import os
import time
import numpy as np
import tensorflow as tf

'''
2018-10-10  Yonv1943
2018-11-15  def update_npz()
2018-11-16  tf_config
'''

np.random.seed(1943)


def save_npy(sess, model_npz):
    timer = time.time()
    tf_vars = tf.global_variables()

    '''save as singal npy'''
    npy_dict = dict()
    for var in tf_vars:
        npy_dict[var.name] = var.eval(session=sess)
        # print("| FETCH: %s" % var.name) if print_info else None
    np.savez_compressed(model_npz, **npy_dict)
    with open(model_npz + '.txt', 'w') as f:
        f.writelines(["%s\n" % key for key in npy_dict.keys()])

    # '''save as several npy'''
    # shutil.rmtree(C.model_npy, ignore_errors=True)
    # os.makedirs(C.model_npy, exist_ok=True)
    # for v in tf_vars:
    #     v_name = str(v.name).replace('/', '-').replace(':', '.') + '.npy'
    #     np.save(os.path.join(C.model_npy, v_name), v.eval(session=sess))
    #     print("| SAVE %s.npy" % v.name) if print_info else None
    print("  SAVE: %s ||Used: %i sec" % (model_npz, time.time() - timer))


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
    np.savez_compressed(dst_path, **dst_ary)
    print("  Update:", dst_path)


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
        print("  Load from npz:", model_npz)


def get_sess(c):
    tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=c.gpu_limit))
    tf_config.gpu_options.allow_growth = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(c.gpu_id)  # choose GPU:0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore the TensorFlow log messages
    sess = tf.Session(config=tf_config)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # recover the TensorFlow log messages
    return sess


def get_saver_logger(c, sess):
    saver = tf.train.Saver(max_to_keep=4)
    if os.path.exists(os.path.join(c.model_dir, 'checkpoint')):
        # C.model_path = tf.train.latest_checkpoint(C.model_dir)
        saver.restore(sess, c.model_path)
        print("| Load from checkpoint:", c.model_path)
    elif os.path.exists(c.model_npz):
        name2ary = np.load(c.model_npz)
        print("| Load from mod.npz")
        sess.run(tf.global_variables_initializer())
        for var_node in tf.global_variables():
            var_ary = name2ary[var_node.name]
            sess.run(tf.assign(var_node, var_ary))
    else:  # Initialize
        os.makedirs(c.model_dir, exist_ok=True)
        sess.run(tf.global_variables_initializer())
        print("| Init:", c.model_path)

    logger = open(c.model_log, 'a')
    previous_train_epoch = sum(1 for _ in open(c.model_log)) if os.path.exists(c.model_log) else 0
    print('  Train_epoch: %6d+%6d' % (previous_train_epoch, c.train_epoch))
    print('  Batch_epoch: %6dx%6d' % (c.batch_epoch, c.batch_size))
    return saver, logger, previous_train_epoch


def draw_plot(log_txt_path):
    print("||" + draw_plot.__name__)
    if not os.path.getsize(log_txt_path):
        print("| NotExist or NullFile:", log_txt_path)
        return None

    arys = np.loadtxt(log_txt_path)
    if arys.shape[0] < 2:
        print("| Empty:", log_txt_path)
        return None

    if 'plt' not in globals():
        import matplotlib.pyplot as plt_global
        global plt
        plt = plt_global

    arys_len = int(len(arys) * 0.9)
    arys = arys[-arys_len:]
    arys = arys.reshape((arys_len, -1, 2)).transpose((1, 0, 2))

    lines = []
    x_pts = np.arange(arys.shape[1])
    x_pts *= arys.shape[0]  # stagger

    # for idx, ary in enumerate(arys[::arys_len // 128]):
    for idx, ary in enumerate(arys):
        x_pts += idx
        y_pts = ary[:, 0]
        e_pts = ary[:, 1]

        y_max = y_pts.max() + 2 ** -32
        y_pts /= y_max
        e_pts /= y_max
        e_pts = np.clip(e_pts, 0, 0.5)
        print("| ymax:", y_max)

        lines.append(plt.plot(x_pts, y_pts, linestyle='dashed', marker='x', markersize=3,
                              label='loss %d, max: %3.0f' % (idx, y_max))[0])
        plt.errorbar(x_pts, y_pts, e_pts, linestyle='None')
    plt.legend(lines, loc='upper right')
    plt.show()
