import cv2
import numpy as np

from configure import Config
from utils import img_util

"""
Enter aerial images taken at different times in the same place,
and select the thinest area of the cloud to replace the original image.
2018-06-30 Source: Yonv1943
2018-10-30 timeline replace
2018-11-28 Removing Clouds and Recovering Ground Observations in Satellite Image Sequences via Temporally Contiguous Robust Matrix Completion
"""

C = Config


def cloud_detection(aerials, white_threshold, knn_para):
    dark_channels = np.vectorize(lambda r, g, b: min(r, g, b)) \
        (aerials[:, :, :, 0], aerials[:, :, :, 1], aerials[:, :, :, 2])

    masks_non_clouds = np.vectorize(lambda v: int(v < white_threshold)) \
        (dark_channels)
    # masks_non_clouds = masks_non_clouds.astype(np.float32)  # for visualization

    mask_always_white = np.vectorize(lambda v: int(v == 0)) \
        (np.sum(masks_non_clouds, axis=0))

    xy_always_white = list()
    xy_not_always_white = list()
    for i, item0 in enumerate(mask_always_white):
        for j, item1 in enumerate(item0):
            if item1 == 1:
                xy_always_white.append((i, j))
            else:
                xy_not_always_white.append((i, j))

    # print(xy_always_white)

    from sklearn.neighbors import NearestNeighbors
    for i, j in xy_always_white:
        data = aerials[:, i, j, :]
        medians = np.median(data, axis=0)

        # data -= np.matmul(np.ones((len(data), 1)), medians[np.newaxis, :])
        nbrs = NearestNeighbors(n_neighbors=knn_para).fit(data)
        distances, idices = nbrs.kneighbors(medians[np.newaxis, :])
        idices = idices[0]
        masks_non_clouds[idices, i, j] = 1  #
        # masks_non_clouds[idices, i, j] = 0.8  # for visualization

    # '''for visualization'''
    # ground = aerials[0]
    # for img, mask, dark_channel in zip(aerials, masks_non_clouds, dark_channels):
    #     mask = mask.astype(np.float32)
    #     mask = mask[:, :, np.newaxis] * np.ones((1, 1, 3))
    #
    #     ground = ground * (1 - mask) + img * mask
    #
    #     img_show = np.concatenate((img, mask, ground), axis=1)
    #     cv2.imshow('', img_show)
    #     cv2.waitKey(123)

    return masks_non_clouds, xy_not_always_white


def inexact_proximal_gradient(aerials, mask_non_clouds, xy_not_always_white):
    grad_non_clouds = np.sum((aerials[1:] - aerials[:-1]) ** 2, axis=3)
    grad_non_clouds = np.concatenate((grad_non_clouds, np.ones_like(aerials[0:1, :, :, 0])), axis=0)
    grad_non_clouds = grad_non_clouds * mask_non_clouds + np.ones_like(grad_non_clouds) * (1.0 - mask_non_clouds)

    ground = np.copy(aerials[0])
    for i, j in xy_not_always_white:
        beg, end = 0, 0
        for end, grad in enumerate(grad_non_clouds[::-1, i, j]):
            if beg == 0 and grad > 0.3:
                beg = end
            elif grad > 0.3:
                break
        # print(111, beg, end)
        if end > beg:
            ground[i, j] = np.average(aerials[beg:end, i, j, :], axis=0)

    cv2.imshow('', np.concatenate((aerials[0], ground), axis=1))
    cv2.waitKey()


def replace_clouds(aerials, cloud1s):  # timeline_replace
    out_aer = aerials[0]
    out_cld = cloud1s[0]
    repeat3 = np.ones([1, 1, 3])

    cloud1s = np.array(cloud1s, np.float32) / 255.0
    cloud1s = np.clip(cloud1s, 0.0, 1.0)

    for aerial, cloud1 in zip(aerials, cloud1s):
        mask01 = cloud1 - out_cld
        mask01[mask01 > 0] = 1.0
        mask01[mask01 < 0] = 0.0
        mask10 = 1.0 - mask01

        out_cld = out_cld * mask01 + cloud1 * mask10
        out_aer = out_aer * mask01[:, :, np.newaxis] * repeat3 + \
                  aerial * mask10[:, :, np.newaxis] * repeat3

        cv2.imshow('beta', np.concatenate((aerial, out_aer), axis=1))
        cv2.waitKey(123)
    out_aer = (out_aer * 255).astype(np.uint8)
    out_cld = (out_cld * 255).astype(np.uint8)
    return out_aer, out_cld


def run_replace():
    ground_path = '/mnt/sdb1/data_sets/AerialImageDataset/test/tyrol-e24.tif'
    ground = cv2.imread(ground_path)  # shape == (5000, 5000, 3)
    # high, width = 1060, 1920
    high, width = 512, 512
    # high, width = 200, 150
    # x_beg, y_beg = 3500, 3200
    x_beg, y_beg = 3600, 3200
    ground = ground[x_beg:x_beg + high, y_beg:y_beg + width]

    cloud1s = img_util.get_cloud1_continusly(1943, 1943 + 8, 1)
    cloud1s = np.array(cloud1s, np.int)[:, :high, :width]
    cloud1s = np.clip((cloud1s - 192) * 8, 0, 255)
    aerials = img_util.get_aerial_continusly(ground, cloud1s)
    aerials = np.array(aerials)

    cv2.imwrite('eval_repkace_aerials.jpg', (aerials[0] * 255).astype(np.uint8))
    out_aer, out_cld = replace_clouds(aerials, cloud1s)
    for img, name in zip([ground, out_aer, out_cld], ['ground', 'out_aer', 'out_cld']):
        cv2.imwrite('eval_replace_%s.jpg' % name, img)

    # mask_non_clouds, xy_not_always_white = cloud_detection(aerials, 0.6, 8)
    # inexact_proximal_gradient(aerials, mask_non_clouds, xy_not_always_white)
    # cv2.waitKey()


def run_eval():
    mat_list = list()
    for name in ['ground', 'out_aer', 'out_cld']:
        img = cv2.imread('eval_replace_%s.jpg' % name)
        mat_list.append(img[np.newaxis, :, :, :])

    import tensorflow as tf
    import mod_mend_buff as mod
    import os
    class Config(object):
        train_epoch = 2 ** 14
        train_size = int(2 ** 17 * 1.9)
        eval_size = 2 ** 3
        batch_size = 2 ** 4
        batch_epoch = train_size // batch_size

        size = int(2 ** 9)  # size = int(2 ** 7)
        replace_num = int(0.25 * batch_size)
        learning_rate = 1e-5  # 1e-4

        show_gap = 2 ** 5  # time
        eval_gap = 2 ** 9  # time
        gpu_limit = 0.48  # 0.0 ~ 1.0
        gpu_id = 1

        data_dir = '/mnt/sdb1/data_sets'
        aerial_dir = os.path.join(data_dir, 'AerialImageDataset/train')
        cloud_dir = os.path.join(data_dir, 'ftp.nnvl.noaa.gov_color_IR_2018')
        grey_dir = os.path.join(data_dir, 'CloudGreyDataset')

        def __init__(self, model_dir='mod'):
            self.model_dir = model_dir
            self.model_name = 'mod'
            self.model_path = os.path.join(self.model_dir, self.model_name)
            self.model_npz = os.path.join(self.model_dir, self.model_name + '.npz')
            self.model_log = os.path.join(self.model_dir, 'training_npy.txt')

    C = Config('mod_mend_GAN_buff')

    gene_name = 'gene'
    inp_ground = tf.placeholder(tf.uint8, [None, C.size, C.size, 3])
    inp_cloud1 = tf.placeholder(tf.uint8, [None, C.size, C.size, 1])

    flt_ground = tf.to_float(inp_ground) / 255.0
    flt_cloud1 = tf.to_float(inp_cloud1) / 255.0
    ten_repeat = tf.ones([1, 1, 1, 3])

    ten_ground = flt_ground[:C.batch_size]
    # buf_ground = flt_ground[C.batch_size:]
    ten_cloud1 = flt_cloud1[:C.batch_size]

    ten_cloud3 = ten_cloud1 * ten_repeat
    ten_mask10 = (1.0 - ten_cloud3)
    ten_ragged = ten_ground * ten_mask10

    ten_patch3 = mod.auto_encoder(tf.concat((ten_ragged, ten_cloud3), axis=3),
                                  32, 3, gene_name, reuse=False)
    out_ground = ten_ragged + ten_patch3 * ten_cloud3

    from utils import mod_util
    sess = mod_util.get_sess(C)
    saver, logger, pre_epoch = mod_util.get_saver_logger(C, sess)
    print("||Training Check")
    # eval_fetch = [ten_ground, out_ground, ten_patch3, ten_cloud3]
    eval_fetch = [out_ground, ten_patch3]
    eval_feed_dict = {inp_ground: mat_list[0],
                      inp_cloud1: mat_list[2][:, :, :, 0:1]}

    mat_list = sess.run(eval_fetch, eval_feed_dict)
    for img, name in zip(mat_list, ['out_ground', 'ten_patch3']):
        img = (img[0] * 255).astype(np.uint8)
        cv2.imshow('beta', img)
        cv2.waitKey(4321)
        print(img.shape, np.max(img))
        cv2.imwrite('eval_gan_%s.jpg' % name, img)

    print(end="  EVAL")


def run():
    # run_replace()
    # run_eval()
    a_org = cv2.imread('eval_replace_ground.jpg')
    a_rep = cv2.imread('eval_replace_out_aer.jpg')
    a_dic = cv2.imread('eval_dict.jpg')
    a_gan = cv2.imread('eval_gan_out_ground.jpg')

    a_mask = cv2.imread('eval_replace_out_cld.jpg')
    a_mask[a_mask > 1] = 1
    mask_num = np.sum(a_mask)
    print("| mask num:", mask_num)
    for a in (a_rep, a_dic, a_gan):
        print("|", np.sum((a_org / 255.0 - a / 255.0) ** 2 / (a_org / 255.0)) / mask_num)
    """
    | mask num: 96336 (512*512)
    | 0.40593090817693867
    | 0.03199842239875413
    | 0.007160325854939165
    """