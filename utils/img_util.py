import os
import glob
import time

import cv2
import numpy as np
import numpy.random as rd
from configure import Config

C = Config()
"""
2018-09-18  23:23:23 fix bug: img_grid() 
2018-09-19  upgrade: img_grid(), random cut
2018-10-21  'class Tools' move from mod_*.py to util.img_util.py 
2018-10-21  poly, blur debug
2018-10-24  stagger plot
2018-11-06  get_data__circle, circle_radius
2018-11-14  image_we_have
"""


class Tools(object):
    def img_check(self, img):
        print("| min,max %6.2f %6.2f |%s" % (np.min(img), np.max(img), img.shape))

    def ary_check(self, ary):
        print("| min,max %6.2f %6.2f |ave,std %6.2f %6.2f |%s" %
              (np.min(ary), np.max(ary), np.average(ary), float(np.std(ary)), ary.shape,))

    def draw_plot(self, log_txt_path):
        print("||" + self.draw_plot.__name__)
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
        for idx, ary in enumerate(arys):
            x_pts += idx
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

    def eval_and_get_img(self, mat_list, img_path, channel):
        mats = np.concatenate(mat_list, axis=3)
        mats = np.clip(mats, 0.0, 1.0)

        out = []
        for mat in mats:
            mat = mat.reshape((C.size, C.size, -1, channel))
            mat = mat.transpose((2, 0, 1, 3))
            mat = np.concatenate(mat, axis=0)
            mat = (mat * 255.0).astype(np.uint8)
            out.append(mat)

        img = np.concatenate(out, axis=1)
        # img = np.rot90(img)
        cv2.imwrite(img_path, img)


class Cloud2Grey(object):
    def __init__(self):
        self.map_pts = np.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'white_line_eliminate_map.npy'))

    def run(self, img):
        for i, j, x, y in self.map_pts:  # eliminate white line
            img[i, j] = img[x, y]

        switch = np.array(img[:, :, 2], dtype=np.int) - img[:, :, 0]
        switch = np.clip(switch - 4, 0, 1)  # The part with thick cloud, could not see the ground

        out = np.clip(img[:, :, 1], 60, 195)
        gray = out - 60
        green = 60 - out
        out = gray * (1 - switch) + green * switch
        return out.astype(np.uint8)


cloud2grey = Cloud2Grey()  # cloud2grey and save as npz


def ary_check(ary):
    ary = np.array(ary)
    print("  min,max %6.2f %6.2f |ave,std %6.2f %6.2f |%s" %
          (ary.min(), ary.max(), np.average(ary), float(np.std(ary)), ary.shape,))


def cover_cloud_mask(img, cloud):
    # cloud = np.clip(cloud * 1.5, 0, 255)  # cloud thickness
    cloud = cloud[:, :, np.newaxis].repeat(3, axis=2)  # channel 1 to channel 3
    img = img * (1 - cloud / 255.0) + cloud
    return img.astype(np.uint8)


def img_grid(img, channel=3):
    xlen, ylen = img.shape[0:2]
    xmod = xlen % C.size
    ymod = ylen % C.size

    xrnd = int(rd.rand() * xmod)
    yrnd = int(rd.rand() * ymod)

    img = img[xrnd:xrnd - xmod, yrnd:yrnd - ymod]  # cut img

    imgs = img.reshape((-1, C.size, ylen // C.size, C.size, channel))
    imgs = imgs.transpose((0, 2, 1, 3, 4))
    imgs = imgs.reshape((-1, C.size, C.size, channel))
    return imgs


def save_cloud_npy(img_path):
    img = cv2.imread(img_path)
    img = cloud2grey.run(img)[:1060, :1920]
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # resize
    img = cv2.blur(img, (3, 3))

    imgs = img_grid(img, channel=1).astype(np.float32)
    imgs = (imgs - imgs.min(axis=(1, 2, 3)).reshape((-1, 1, 1, 1)))
    imgs = imgs * (255.0 / ((imgs.max(axis=(1, 2, 3)) + 1.0).reshape((-1, 1, 1, 1))))

    npy_name = os.path.basename(img_path)[:-4] + '.npz'
    npy_path = os.path.join(C.grey_dir, npy_name)
    np.savez_compressed(npy_path, imgs.astype(np.uint8))


def mp_pool(func, iterable):
    print("  mp_pool iterable: %6d |func: %s" % (len(iterable), func.__name__))
    import multiprocessing as mp

    with mp.Pool(processes=min(16, (mp.cpu_count() - 2) // 2)) as pool:
        res = pool.map(func, iterable)
    return res


def get_data__aerial_imgs(img_path):
    img = cv2.imread(img_path)
    imgs = img_grid(img, channel=3)
    # imgs = imgs.astype(np.float32) / 255.0
    return imgs


def get_data__greysc_imgs(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    imgs = img_grid(img, channel=1)
    # imgs = imgs.astype(np.float32) / 255.0
    return imgs


def get_data__ground(data_size, channel=3):
    img_per_img = int((5000 // C.size) * (5000 // C.size))
    img_we_need = data_size // img_per_img + 1

    img_paths = glob.glob(os.path.join(C.aerial_dir, '*.tif'))
    from random import shuffle
    shuffle(img_paths)
    print('  Image we have: %s  we need: %s' % (len(img_paths), img_we_need))
    img_paths = img_paths[:img_we_need]

    pooling_func = get_data__aerial_imgs if channel == 3 else get_data__greysc_imgs
    data_sets = mp_pool(func=pooling_func, iterable=img_paths)

    data_sets_shape = list(data_sets[0].shape)
    data_sets_shape[0] = -1

    data_set = np.reshape(data_sets, data_sets_shape)
    data_set = data_set[:data_size]

    print("  %s | shape: %s" % (get_data__ground.__name__, data_set.shape))
    return data_set


def get_data__mask01(data_size, channel=1):
    ten = np.zeros([1, C.size, C.size, 1])
    ten[0, int(0.25 * C.size):int(0.75 * C.size), int(0.25 * C.size):int(0.75 * C.size), 0] = 1.0
    ten = ten * np.ones([data_size, 1, 1, 1])
    return ten


def get_data__circle(data_size, circle_num):
    mats = []
    circle_xyrs = rd.randint(0.25 * C.size, 0.75 * C.size, (data_size, circle_num, 3))
    for c123 in circle_xyrs:
        img = np.zeros((C.size, C.size))
        for cx, cy, cr in c123:
            img = cv2.circle(img, (cx, cy), int(cr * 0.75 / circle_num), 1.0, cv2.FILLED)
        # img = cv2.blur(img, (3, 3))[:, :, np.newaxis]  # 1943

        mats.append(img[np.newaxis, :, :, np.newaxis])
    return np.concatenate(mats, axis=0)


def get_data__cloud1_imgs(npy_path):
    imgs = np.load(npy_path)['arr_0']
    # imgs = imgs.astype(np.float32) / 255.0
    return imgs


def get_data__cloud1(data_size):
    """cloud_npy"""
    img_per_npy = int((1060 / 2 // C.size) * (1920 / 2 // C.size))
    img_we_need = data_size // img_per_npy + 1

    img_paths = glob.glob(os.path.join(C.cloud_dir, '*.jpg'))
    img_paths = img_paths[:img_we_need]

    os.makedirs(C.grey_dir, exist_ok=True)
    npy_names = set([p[:-4] for p in os.listdir(C.grey_dir)])
    img_paths = [p for p in img_paths if os.path.basename(p)[:-4] not in npy_names]
    mp_pool(func=save_cloud_npy, iterable=img_paths)

    """cloud_data_set"""
    npy_paths = glob.glob(os.path.join(C.grey_dir, '*.npz'))[:img_we_need]
    data_sets = mp_pool(func=get_data__cloud1_imgs, iterable=npy_paths)

    data_sets_shape = list(data_sets[0].shape)
    data_sets_shape[0] = -1

    data_set = np.reshape(data_sets, data_sets_shape)
    data_set = data_set[:data_size]

    print("  %s | shape: %s" % (get_data__cloud1.__name__, data_set.shape))
    return data_set


def get_cloud1_continusly(beg_idx, end_idx, step):
    cloud2grey = Cloud2Grey()  # cloud2grey and save as npz
    img_paths = glob.glob(os.path.join(C.cloud_dir, '*.jpg'))
    img_paths = sorted(img_paths)[beg_idx:end_idx:step]
    imgs = list()
    for img_path in img_paths:
        cloud1 = cv2.imread(img_path)
        cloud1 = cloud2grey.run(cloud1)[:1060, :1920]
        cloud1 = cv2.blur(cloud1, (3, 3))
        cloud1 = np.clip(cloud1, 0, 127) * 2

        imgs.append(cloud1)
        # cv2.imshow('beta', cloud1)
        # cv2.waitKey(234)

    return imgs  # shape(w, h), dtype(np.uint8)(0, 255)


def get_aerial_continusly(ground, cloud1s):
    imgs = list()
    for cloud1 in cloud1s:
        img = cover_cloud_mask(ground, cloud1)
        img = img.astype(np.float32) / 255.0
        imgs.append(img)

    return imgs  # shape(w, h, 3), dtype(float32)(0.0, 1.0)


def test():
    cv2.namedWindow('beta', cv2.WINDOW_KEEPRATIO)
    img = cv2.imread(os.path.join(C.aerial_dir, 'bellingham1.tif'))
    cv2.imshow('beta', img)
    cv2.waitKey(3456)


def get_eval_img(mat_list, img_path, channel):
    mats = np.concatenate(mat_list, axis=3)
    mats = np.clip(mats, 0.0, 1.0)
    out = []
    for mat in mats:
        mat = mat.reshape((C.size, C.size, -1, channel))
        mat = mat.transpose((2, 0, 1, 3))
        mat = np.concatenate(mat, axis=0)
        mat = (mat * 255.0).astype(np.uint8)
        out.append(mat)

    img = np.concatenate(out, axis=1)
    cv2.imwrite(img_path, img)


