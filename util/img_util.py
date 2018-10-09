import os
import glob

import cv2
import numpy as np
import numpy.random as rd
from configure import Config

G = Config()
"""
2018-09-18 23:23:23 fix bug: img_grid() 
2018-09-19 15:12:12 upgrade: img_grid(), random cut
"""


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
    xmod = xlen % G.size
    ymod = ylen % G.size

    xrnd = int(rd.rand() * xmod)
    yrnd = int(rd.rand() * ymod)

    img = img[xrnd:xrnd - xmod, yrnd:yrnd - ymod]  # cut img

    imgs = img.reshape((-1, G.size, ylen // G.size, G.size, channel))
    imgs = imgs.transpose((0, 2, 1, 3, 4))
    imgs = imgs.reshape((-1, G.size, G.size, channel))
    return imgs


def save_cloud_npy(img_path):
    img = cv2.imread(img_path)
    img = cloud2grey.run(img)[:1060, :1920]
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # resize
    img = cv2.blur(img, (5, 5))

    imgs = img_grid(img, channel=1).astype(np.float32)
    imgs = (imgs - imgs.min(axis=(1, 2, 3)).reshape((-1, 1, 1, 1)))
    imgs = imgs * (255.0 / ((imgs.max(axis=(1, 2, 3)) + 1.0).reshape((-1, 1, 1, 1))))

    npy_name = os.path.basename(img_path)[:-4] + '.npz'
    npy_path = os.path.join(G.grey_dir, npy_name)
    np.savez_compressed(npy_path, imgs.astype(np.uint8))


def get_imgs_for_cloud_data_set(npy_path):
    imgs = np.load(npy_path)['arr_0']
    imgs = imgs.astype(np.float32) / 255.0
    return imgs


def get_imgs_for_aerial_data_set(img_path):
    img = cv2.imread(img_path)
    imgs = img_grid(img, channel=3)
    imgs = imgs.astype(np.float32) / 255.0
    return imgs


def mp_pool(func, iterable):
    print("  mp_pool iterable: %6d |func: %s" % (len(iterable), func.__name__))
    import multiprocessing as mp
    with mp.Pool(processes=min(16, mp.cpu_count() - 2)) as pool:
        res = pool.map(func, iterable)
    return res


def get_data_sets(data_size):
    print("| %s" % get_data_sets.__name__)

    """aerial_data_set"""
    img_per_img = int((5000 // G.size) * (5000 // G.size))
    img_we_need = data_size // img_per_img + 1

    img_paths = glob.glob(os.path.join(G.aerial_dir, '*.tif'))[:img_we_need]
    data_sets = mp_pool(func=get_imgs_for_aerial_data_set, iterable=img_paths)
    aerial_data_set = []
    for data_set in data_sets:
        aerial_data_set.extend(data_set)
    aerial_data_set = np.array(aerial_data_set[:data_size])

    """cloud_npy"""
    img_per_npy = int((1060 / 2 // G.size) * (1920 / 2 // G.size))
    img_we_need = data_size // img_per_npy + 1

    img_paths = glob.glob(os.path.join(G.cloud_dir, '*.jpg'))
    img_paths = img_paths[:img_we_need]

    os.makedirs(G.grey_dir, exist_ok=True)
    npy_names = set([p[:-4] for p in os.listdir(G.grey_dir)])
    img_paths = [p for p in img_paths if os.path.basename(p)[:-4] not in npy_names]
    mp_pool(func=save_cloud_npy, iterable=img_paths)

    """cloud_data_set"""
    npy_paths = glob.glob(os.path.join(G.grey_dir, '*.npz'))[:img_we_need]
    data_sets = mp_pool(func=get_imgs_for_cloud_data_set, iterable=npy_paths)
    cloud_data_set = []
    for data_set in data_sets:
        cloud_data_set.extend(data_set)
    cloud_data_set = np.array(cloud_data_set[:data_size])

    '''data_sets'''
    data_sets = (aerial_data_set, cloud_data_set)
    for ary in data_sets:
        print(end='  len %d |' % len(ary))
        ary_check(ary[0])
    return data_sets


if __name__ == '__main__':
    get_data_sets(data_size=4)
