import cv2
import numpy as np

from configure import Config
from utils import img_util
from utils import mod_util

"""
Enter aerial images taken at different times in the same place,
and select the thinest area of the cloud to replace the original image.
2018-06-30 Source: Yonv1943
2018-10-30 timeline replace
2018-11-28 Removing Clouds and Recovering Ground Observations in Satellite Image Sequences via Temporally Contiguous Robust Matrix Completion
"""


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

    ten_patch3 = mod.generator(tf.concat((ten_ragged, ten_cloud3), axis=3),
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


def run_evaluation_metrics():
    # run_replace()
    # run_eval()
    a_org = cv2.imread('eval_replace_ground.jpg')
    a_rep = cv2.imread('eval_replace_out_aer.jpg')
    a_dic = cv2.imread('eval_dict.jpg')
    a_gan = cv2.imread('eval_gan_out_ground.jpg')

    a_mask = cv2.imread('eval_replace_out_cld.jpg')
    a_mask[a_mask > 1] = 1

    from skimage.measure import compare_psnr as psnr
    from skimage.measure import compare_mse as mse
    from skimage.measure import compare_ssim as ssim

    for a in (a_rep, a_dic, a_gan):  # ssim
        print("| PSNR %e MSE %e SSIM: %e" % (psnr(a_org, a),
                                             mse(a_org, a),
                                             ssim(a_org, a, multichannel=True)))
    # a_mask = cv2.imread('eval_replace_out_cld.jpg')
    # print(np.sum(a_mask // 255), a_mask.shape[0] * a_mask.shape[1])
    # 11205 262144
    """
    | PSNR 1.805008e+01 MSE 1.018760e+03 SSIM: 8.988292e-01
    | PSNR 2.858750e+01 MSE 9.001815e+01 SSIM: 9.069278e-01
    | PSNR 3.486142e+01 MSE 2.122945e+01 SSIM: 9.710586e-01
    """
    # mask_num = np.sum(a_mask)
    # print("| mask num:", mask_num)
    # for a in (a_rep, a_dic, a_gan):  # EER
    #     print("|", np.sum((a_org / 255.0 - a / 255.0) ** 2 / (a_org / 255.0)) / mask_num)
    """
    | mask num: 96336 (512*512)
    | 0.40593090817693867
    | 0.03199842239875413
    | 0.007160325854939165
    """


def auto_canny(img, offset=0.25):
    lower = np.percentile(img, int((0.5 - offset) * 100))
    upper = np.percentile(img, int((0.5 + offset) * 100))
    # print(lower, upper)  # 69, 131
    edged = cv2.Canny(img, lower, upper)
    # edged = cv2.Canny(img, 69, 131)
    return edged


def run_draw_canny():
    # img = cv2.imread('occlusion-of-clouds.png')
    # img = img / 255
    # print(img.shape)
    #
    # img = (img * 255).astype(np.uint8)
    #
    # mask = np.arange(768, 0, -1) / 768
    # mask = mask.reshape((1, 768, 1))
    # mask = mask * np.ones_like(img)
    # # mask[400:] = 0
    #
    # hist_edge = np.empty((768, 768))
    # for i in range(img.shape[1]):
    #     img = np.concatenate((img[:, 1:], img[:, 0:1]), axis=1)
    #
    #     edge = img / 255
    #     edge = edge * (1 - mask) + mask
    #     edge = (edge * 255).astype(np.uint8)
    #     edge = auto_canny(edge)
    #
    #     sum_edge = np.sum(edge, axis=0)
    #     hist_edge[i] = sum_edge
    #     # cv2.imshow('cv2', edge)
    #     # cv2.waitKey(1)
    # np.save('hist_edge.npy', hist_edge)
    hist_edge = np.load('hist_edge.npy')
    hist_edge = np.sum(hist_edge, axis=0)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots()
    axs.plot(np.arange(len(hist_edge)), hist_edge)
    plt.show()

    # mask = np.arange(768, 0, -1) / 768
    # mask = mask.reshape((1, 768, 1))
    # mask = mask * np.ones_like(img)
    # mask[400:] = 0
    #
    # img = img * (1 - mask) + mask
    # img = (img * 255).astype(np.uint8)
    # cv2.imwrite('cloud-gradient-fill.png', img)
    #
    # img = auto_canny(img)
    # cv2.imwrite('auto-canny.png', img)
    # # cv2.imshow('', img)
    # # cv2.waitKey(5432)


def run_eval_haze():
    img = cv2.imread('road-thin.png')[np.newaxis, :, :, :]
    img = np.pad(img, ((0, 0), (32, 32), (32, 32), (0, 0)), 'reflect')
    eval_list = [img, np.zeros_like(img[:, :, :, 0:1])]
    from mod_haze_unet import init_train
    inp_ground, inp_mask01, train_fetch, eval_fetch = init_train()

    C = Config('mod_haze_unet')
    C.size = img.shape[1]
    sess = mod_util.get_sess(C)
    mod_util.get_saver_logger(C, sess)
    print("||Training Check")
    eval_feed_dict = {inp_ground: eval_list[0],
                      inp_mask01: eval_list[1], }
    img_util.get_eval_img(mat_list=sess.run(eval_fetch, eval_feed_dict), channel=3,
                          img_path="%s/eval-%08d.jpg" % ('temp', 0))


def run_eval_mend():
    img = cv2.imread('road-car.png')[np.newaxis, :, :, :]
    img = np.pad(img, ((0, 0), (32, 32), (32, 32), (0, 0)), 'reflect')
    # mask = cv2.imread('road-label.png')[np.newaxis, :, :, :]
    mask = cv2.imread('road-cloud0.png')[np.newaxis, :, :, :]
    mask = np.pad(mask, ((0, 0), (32, 32), (32, 32), (0, 0)), 'reflect')[:, :, :, 0:1]

    threshold = 244
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255

    # cv2.imshow('', mask[0])
    # cv2.waitKey(5432)
    eval_list = [img, mask, img, mask]

    from mod_mend_dila import init_train
    C = Config('mod_mend_dila')
    from mod_mend_nres import init_train
    C = Config('mod_mend_nres')
    inp_ground, inp_mask01, inp_grdbuf, inp_mskbuf, fetch, eval_fetch = init_train()

    C.size = img.shape[1]
    sess = mod_util.get_sess(C)
    mod_util.get_saver_logger(C, sess)
    print("||Training Check")
    eval_feed_dict = {inp_ground: eval_list[0],
                      inp_mask01: eval_list[1],
                      inp_grdbuf: eval_list[2],
                      inp_mskbuf: eval_list[3], }
    img_util.get_eval_img(mat_list=sess.run(eval_fetch, eval_feed_dict), channel=3,
                          img_path="%s/eval-%08d.jpg" % ('temp', 0))



def run():
    # run_evaluation_metrics()
    # run_draw_canny()
    # run_eval_haze()
    run_eval_mend()
