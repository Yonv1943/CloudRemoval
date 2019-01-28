import os
import glob
import time

import cv2
import numpy as np
import numpy.random as rd
from itertools import product as iter_product
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

        thold = np.array(img[:, :, 2], dtype=np.int) - img[:, :, 0]

        # thold = np.heaviside(thold - 4, 0)
        thold = np.clip(thold - 4, 0, 1)  # The part with thick cloud, could not see the ground

        out = np.clip(img[:, :, 1], 60, 195)
        gray = out - 60
        green = 60 - out
        out = gray * (1 - thold) + green * thold
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


def img_grid(img, size, random_cut=True):
    # def img_grid_reverse(imgs, x_num, y_num):
    #     x_size, y_size, channel = imgs.shape[-3:]
    #     imgs = imgs.reshape((x_num, y_num, x_size, y_size, channel))
    #     imgs = imgs.transpose((0, 2, 1, 3, 4))
    #     img = imgs.reshape((x_num * x_size, y_num * y_size, channel))
    #     return img
    """
    x_len, y_len
    random_cut
    """
    x_len, y_len, channel = img.shape[0:3]
    x_mod = x_len % size
    y_mod = y_len % size
    # print("%s:" % img_grid.__name__, x_len // size, y_len // size, size)  # for img_grid_reverse()

    if random_cut:
        x_beg = int(rd.rand() * x_mod)
        x_end = x_beg - x_mod if x_beg != x_mod else None
        y_beg = int(rd.rand() * y_mod)
        y_end = y_beg - y_mod if y_beg != y_mod else None

        img = img[x_beg:x_end, y_beg:y_end]  # cut img
    else:
        img = img[x_mod:, y_mod:]

    imgs = img.reshape((-1, size, y_len // size, size, channel))
    imgs = imgs.transpose((0, 2, 1, 3, 4))
    imgs = imgs.reshape((-1, size, size, channel))
    return imgs


def slide_window(img, size, pads):
    len0, len1 = img.shape[0:2]

    num0 = len0 // size
    num1 = len1 // size
    num0 = num0 if len0 % size == 0 else num0 + 1
    num1 = num1 if len1 % size == 0 else num1 + 1

    img = np.pad(img, ((pads, pads + num0 * size),
                       (pads, pads + num1 * size), (0, 0)), 'reflect')

    grids = list()
    win_size = size + 2 * pads
    for i, j in iter_product(range(0, num0 * size, size),
                             range(0, num1 * size, size)):
        grid = img[i:i + win_size, j:j + win_size]
        grids.append(grid)
    grids = np.stack(grids, axis=0)  # grids = np.concatenate(grids[np.newaxis, :, :, :], axis=0)
    # slide_info = (size, pads, len0, len1, num0, num1)
    # return grids, slide_info
    return grids


def slide_window_reverse(grids, info):
    (size, pads, len0, len1, num0, num1) = info
    img = np.empty((size * num0,
                    size * num1, 3), grids.dtype)

    iter_grid = iter(grids[:, pads:pads + size, pads:pads + size])
    for i, j in iter_product(range(0, num0 * size, size),
                             range(0, num1 * size, size)):
        img[i:i + size, j:j + size] = next(iter_grid)
    img = img[:len0, :len1]
    return img


def img_show(img, wait_key=1, win_name='Check'):
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imshow(win_name, img)
    cv2.waitKey(wait_key)


def save_cloud_npy(img_path):
    img = cv2.imread(img_path)
    img = cloud2grey.run(img)[:1060, :1920]
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # resize
    img = cv2.blur(img, (3, 3))

    imgs = img_grid(img, C.size).astype(np.float32)
    imgs = (imgs - imgs.min(axis=(1, 2, 3)).reshape((-1, 1, 1, 1)))
    imgs = imgs * (255.0 / ((imgs.max(axis=(1, 2, 3)) + 1.0).reshape((-1, 1, 1, 1))))

    npy_name = os.path.basename(img_path)[:-4] + '.npz'
    npy_path = os.path.join(C.grey_dir, npy_name)
    np.savez_compressed(npy_path, imgs.astype(np.uint8))


def mp_pool(func, iterable):
    # print("  mp_pool iterable: %6d |func: %s" % (len(iterable), func.__name__))
    import multiprocessing as mp

    with mp.Pool(processes=min(16, (mp.cpu_count() - 2) // 2)) as pool:
        res = pool.map(func, iterable)
    return res


def get_data__aerial_imgs(img_path, pads=8):
    img = cv2.imread(img_path)
    # imgs = img_grid(img, C.size)

    size = C.size - pads * 2
    imgs = slide_window(img, size, pads)
    # print(len(imgs))
    return imgs


def get_data__cloud1_imgs(npy_path):
    imgs = np.load(npy_path)['arr_0']
    # imgs = imgs.astype(np.float32) / 255.0
    return imgs


def get_data__ground(data_size):
    img_per_img = len(slide_window(np.empty((5000, 5000, 3), np.uint8), C.size - 8 * 2, 8))  # pads=8
    img_we_need = data_size // img_per_img + 1

    img_paths = glob.glob(os.path.join(C.aerial_dir, '*.tif'))
    print('  img we have: %s  we need: %s' % (len(img_paths), img_we_need))
    if img_we_need > len(img_paths):
        aerial_test = os.path.join(C.data_dir, 'AerialImageDataset/test')
        img_paths.extend(glob.glob(os.path.join(aerial_test, '*.tif')))
        print('  Image we have: %s  we need: %s' % (len(img_paths), img_we_need))

    img_paths = img_paths[:img_we_need]

    pooling_func = get_data__aerial_imgs
    data_sets = mp_pool(func=pooling_func, iterable=img_paths)

    data_sets_shape = list(data_sets[0].shape)
    data_sets_shape[0] = -1

    data_set = np.reshape(data_sets, data_sets_shape)
    data_set = data_set[:data_size]

    # print("  %s | shape: %s" % (get_data__ground.__name__, data_set.shape))
    return data_set


def get_data__mask01(data_size, channel=1):
    ten = np.zeros([1, C.size, C.size, 1])
    ten[0, int(0.25 * C.size):int(0.75 * C.size), int(0.25 * C.size):int(0.75 * C.size), 0] = 1.0
    ten = ten * np.ones([data_size, 1, 1, 1])
    return ten


def get_data__circle(data_size, circle_num, radius_rate=0.25):
    mats = []
    circle_xyrs = rd.randint(0.25 * C.size, 0.75 * C.size, (data_size, circle_num, 3))
    for c123 in circle_xyrs:
        img = np.zeros((C.size, C.size), np.uint8)
        for cx, cy, cr in c123:
            img = cv2.circle(img, (cx, cy), int(cr * radius_rate), 255, cv2.FILLED)
        # img = cv2.blur(img, (3, 3))[:, :, np.newaxis]  # 1943

        mats.append(img[np.newaxis, :, :, np.newaxis])
    return np.concatenate(mats, axis=0)


def get_data__cloud1(data_size):
    img_per_npy = int((1060 / 2 // C.size) * (1920 / 2 // C.size))
    img_we_need = data_size // img_per_npy + 1

    """cloud_data_set"""
    npy_paths = glob.glob(os.path.join(C.grey_dir, '*.npz'))
    print('  npy we have: %s  we need: %s' % (len(npy_paths), img_we_need))
    if img_we_need > len(npy_paths):
        img_paths = glob.glob(os.path.join(C.cloud_dir, '*.jpg'))

        if img_we_need > len(img_paths):
            cloud_dir_2017 = os.path.join(C.data_dir, 'ftp.nnvl.noaa.gov_color_IR_2017')
            img_paths.extend(glob.glob(os.path.join(cloud_dir_2017, '*.jpg')))
        img_paths = img_paths[:img_we_need]
        print('img', len(img_paths))

        """cloud_npy"""
        os.makedirs(C.grey_dir, exist_ok=True)
        npy_names = set([p[:-4] for p in os.listdir(C.grey_dir)])
        img_paths = [p for p in img_paths if os.path.basename(p)[:-4] not in npy_names]
        mp_pool(func=save_cloud_npy, iterable=img_paths)
        npy_paths = glob.glob(os.path.join(C.grey_dir, '*.npz'))
        print('  npy we have: %s  we need: %s' % (len(npy_paths), img_we_need))
    npy_paths = npy_paths[:img_we_need]

    data_sets = mp_pool(func=get_data__cloud1_imgs, iterable=npy_paths)
    data_sets_shape = list(data_sets[0].shape)
    data_sets_shape[0] = -1

    data_set = np.reshape(data_sets, data_sets_shape)

    if len(data_set) > data_size:
        data_set = data_set[:data_size]
    else:  # add
        data_set = np.concatenate((data_set, data_set[:data_size - len(data_set)]))

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


def get_eval_img(mat_list, img_path, channel, img_write=True):
    # [print(mat.shape) for mat in mat_list]
    ary_repeat = np.ones((1, 1, 1, 3))
    for i in range(len(mat_list)):
        if mat_list[i].shape[3] != 3:
            mat_list[i] = mat_list[i] * ary_repeat

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

    if img_write:
        cv2.imwrite(img_path, img)
    else:
        cv2.imshow('', img)
        cv2.waitKey(1)
