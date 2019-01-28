import os
import os.path as op
import glob
import cv2
import numpy as np
import numpy.random as rd
import time


def cloud_detect(aerials):
    import tensorflow as tf
    from configure import Config
    from utils import mod_util
    from mod_cloud_detect import unet
    C = Config('mod_cloud_detect')

    size = aerials.shape[1]

    unet_name, unet_dim = 'unet', 24
    inp_aerial = tf.placeholder(tf.uint8, [None, size, size, 3])
    ten_aerial = tf.to_float(inp_aerial) / 255
    eva_grdcld = unet(ten_aerial, unet_dim, 4, unet_name, reuse=False, training=False)
    eva_ground = eva_grdcld[:, :, :, 0:3]
    eva_mask01 = eva_grdcld[:, :, :, 3:4]

    sess = mod_util.get_sess(C)
    mod_util.get_saver_logger(C, sess)
    print("||Training Check")

    # aerials_shape = list(aerials.shape[-1:])
    # aerials_shape = [-1, 16] + aerials_shape
    # aerials = aerials.reshape(aerials_shape)

    grounds = list()
    mask01s = list()
    for i, aerial in enumerate(aerials):
        eval_feed_dict = {inp_aerial: aerial[np.newaxis, :, :, :]}
        # eval_fetch = [ten_aerial, eva_ground, eva_mask01]
        eval_fetch = [eva_ground, eva_mask01]
        mat_list = sess.run(eval_fetch, eval_feed_dict)

        grounds.append(np.clip(mat_list[0] * 255, 0, 255).astype(np.uint8))
        mask01s.append(np.clip(mat_list[1] * 255, 0, 255).astype(np.uint8))

        # img_util.get_eval_img(mat_list=mat_list,channel=3, img_write=False
        #                                  img_path="%s/eval-%08d.jpg" % ('temp', 0),)
        if rd.rand() < 0.01:
            print('Eval:', i)

    # def mats_list2jpg(mats_list, save_name):
    #     mats = np.concatenate(mats_list, axis=0)
    #     img = img_grid_reverse(mats)
    #     cv2.imwrite(save_name, img)
    #
    # mats_list2jpg(grounds, 'su_zhou/ground.jpg')
    # mats_list2jpg(mask01s, 'su_zhou/mask01.jpg')
    grounds = np.concatenate(grounds, axis=0)
    mask01s = np.concatenate(mask01s, axis=0)
    return grounds, mask01s


def cloud_removal(aerials, label1s):
    import tensorflow as tf
    from configure import Config
    from utils import mod_util
    from mod_cloud_remove_rec import auto_encoder
    C = Config('mod_cloud_remove_rec')

    size = aerials.shape[1]

    gene_name, gene_dim = 'gene', 32
    inp_ground = tf.placeholder(tf.uint8, [None, size, size, 3])
    ten_ground = tf.to_float(inp_ground) / 255
    inp_mask01 = tf.placeholder(tf.uint8, [None, size, size, 1])
    ten_mask01 = tf.to_float(inp_mask01) / 255

    ten_mask10 = (1.0 - ten_mask01)
    ten_ragged = ten_ground * ten_mask10

    ten_patch3 = auto_encoder(ten_ragged - ten_mask01,
                              gene_dim, 3, gene_name,
                              reuse=False, training=False)
    out_ground = ten_ragged + ten_patch3 * ten_mask01

    sess = mod_util.get_sess(C)
    mod_util.get_saver_logger(C, sess)
    print("||Training Check")

    patch3s = list()
    grounds = list()

    for i, (aerial, label1) in enumerate(zip(aerials, label1s)):
        aerial = aerial[np.newaxis, :, :, :]
        label1 = label1[np.newaxis, :, :, 0:1]

        eval_feed_dict = {inp_ground: aerial,
                          inp_mask01: label1, }
        eval_fetch = [ten_patch3, out_ground]
        mat_list = sess.run(eval_fetch, eval_feed_dict)

        patch3s.append(np.clip(mat_list[0] * 255, 0, 255).astype(np.uint8))
        grounds.append(np.clip(mat_list[1] * 255, 0, 255).astype(np.uint8))
        if i % 64 == 0:
            print('Eval:', i)

    grounds = np.concatenate(grounds, axis=0)
    patch3s = np.concatenate(patch3s, axis=0)
    return grounds, patch3s,


def imshow(img, wait_key=1, win_name='Check', ):
    try:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_key)
    except Exception as e:
        print("|| IMG ERROR:", e)

def save(i):
    cv2.imwrite('result/%06d-0-cloud1.png' % i, cloud1s[i])
    cv2.imwrite('result/%06d-1-mask01.png' % i, np.heaviside(cloud1s[i] - np.percentile(cloud1s[i], 85), 0) * 255)
    cv2.imwrite('result/%06d-2-ground.png' % i, grounds[i])
    cv2.imwrite('result/%06d-3-result.png' % i, results[i])

def run_cloud_removal():
    # idxs  = set([int(n[:6]) for n in os.listdir('result')])
    # for i in idxs:
    #     ground = cv2.imread('result/%06d-2-ground.png' % i)
    #     cloud1 = cv2.imread('result/%06d-0-cloud1.png' % i)
    #
    #     thold = np.percentile(cloud1, 85) / 255.0
    #
    #     cloud3 = cloud1 / 255.0
    #     if thold > 0.1:
    #         cloud3 -= thold - 0.1
    #         cloud3 *= 0.6 / 0.1
    #     else:
    #         cloud3 = np.heaviside(cloud3 - thold, 0)
    #         cloud3 = cv2.blur(cloud3, (5, 5))
    #
    #     cloud3 = (np.clip(cloud3, 0, 1) * 255).astype(np.uint8)
    #     cv2.imwrite('result/%06d-0-cloud3.png' % i, cloud3)
    #     # imshow(cloud3, 1234)
    #
    #     mask01 = cloud3 / 255.0
    #     aerial = ground / 255.0
    #     aerial = aerial * (1 - mask01) + mask01
    #     aerial = (np.clip(aerial, 0, 255) * 255).astype(np.uint8)
    #     cv2.imwrite('result/%06d-4-aerial.png' % i, aerial)

    # idxs  = set([int(n[:6]) for n in os.listdir('result')])
    # aerials = list()
    # for i in idxs:
    #     aerial = cv2.imread('result/%06d-4-aerial.png' % i)
    #     aerials.append(aerial)
    # aerials = np.stack(aerials)
    # grounds, cloud1s = cloud_detect(aerials)
    #
    # for i, idx in enumerate(idxs):
    #     cv2.imwrite('result/%06d-4-ground-thin.png' % idx, grounds[i])
    #     cv2.imwrite('result/%06d-4-cloud1-thin.png' % idx, cloud1s[i])



    idxs  = set([int(n[:6]) for n in os.listdir('result')])
    aerials = list()
    mask01s = list()
    dilate_kernel = np.ones((3, 3))
    for i in idxs:
        aerial = cv2.imread('result/%06d-4-ground-thin.png' % i)
        aerials.append(aerial)
        mask01 = cv2.imread('result/%06d-1-mask01.png' % i)
        mask01 = cv2.dilate(mask01, dilate_kernel)
        mask01s.append(mask01)
    aerials = np.stack(aerials)
    mask01s = np.stack(mask01s)
    results, patch3s = cloud_removal(aerials, mask01s)

    for i, idx in enumerate(idxs):
        cv2.imwrite('result/%06d-3-result-thin.png' % idx, results[i])



    # import tensorflow as tf
    #
    # # grounds = np.load('grounds.npy')[:4] / 255.0
    # # cloud1s = np.load('cloud1s.npy')[:4] / 255.0
    # # cloud1s = cloud1s * 2 - 0.75
    # # cloud1s = np.clip(cloud1s, 0, 1)
    # # cloud1s[:, :+4, :, 0] = 0
    # # cloud1s[:, -4:, :, 0] = 0
    # # cloud1s[:, :, :+4, 0] = 0
    # # cloud1s[:, :, -4:, 0] = 0
    #
    # # aerials = grounds * (1 - cloud1s) + cloud1s
    # # aerials = (aerials * 255).astype(np.uint8)
    # # tf.reset_default_graph()
    # # grounds, cloud1s = cloud_detect(aerials)
    #
    # # tf.reset_default_graph()
    # # mask01s = cloud1s / 255.0
    # # mask01s = np.heaviside(mask01s - 0.5, 0)
    # # mask01s = (mask01s * 255).astype(np.uint8)
    #
    # grounds = np.load('grounds.npy')
    # cloud1s = np.load('cloud1s.npy')
    #
    # def get_mask01(mats, percentile=85):
    #     thresholds = np.percentile(mats, percentile, axis=(1, 2, 3), keepdims=True)
    #     mats[mats < thresholds] = 0
    #     mats[mats >= thresholds] = 255
    #     return mats
    # mask01s = get_mask01(cloud1s, 85)
    #
    # grounds, patch3s = cloud_removal(grounds, mask01s)
    #
    # np.save('results.npy', grounds)

    # results = np.load('results.npy')
    # for i in np.random.randint(0, len(results), 256):
    #     cv2.imwrite('temp/%06d.png' % i, results[i])



def run_eval_metrics():
    from skimage.measure import compare_psnr as psnr
    from skimage.measure import compare_mse as mse
    from skimage.measure import compare_ssim as ssim

    grounds = np.load('grounds.npy')

    for dir in ('eval_low_rank', 'eval_dict', 'eval_two_stage1', 'eval_ground'):
        eval_matrics = list()
        for i, ground in enumerate(grounds):
            if i == 3 or 8 <= i <= 10:
                continue
            # ground = cv2.imread('%s/%02d.jpg' % ('eval_ground', i))
            result = cv2.imread('%s/%02d.jpg' % (dir, i))
            # print("| PSNR %e MSE %e SSIM: %e" %
            #       (psnr(ground, result),
            #        mse(ground, result),
            #        ssim(ground, result, multichannel=True)))
            eval_matrics.append((
                ssim(ground, result, multichannel=True),
                psnr(ground, result),
                mse(ground, result),
                np.mean((ground - result) ** 2),
            ))

        eval_matrics = np.array(eval_matrics)
        np.save('%s/eval.npy' % dir, eval_matrics)

    for dir in ('eval_low_rank', 'eval_dict', 'eval_two_stage1', 'eval_ground'):
        print("%16s SSIM PSNR MSE L2:" % dir,
              np.average(np.load('%s/eval.npy' % dir), axis=0))


def run():
    # run_eval_metrics()
    run_cloud_removal()
    pass


if __name__ == '__main__':
    run()
