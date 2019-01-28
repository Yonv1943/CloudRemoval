import os


class Config(object):
    train_epoch = 2 ** 9
    train_size = int(2 ** 10)
    eval_size = 2 ** 3
    batch_size = int(2 ** 4 * 1.25)
    batch_epoch = train_size // batch_size

    size = int(2 ** 8)  # int(2 ** 7)
    replace_num = int(0.368 * batch_size)
    learning_rate = 8e-5  # 1e-4

    show_gap = 2 ** 2  # time
    eval_gap = 2 ** 2  # time
    gpu_limit = 0.9  # 0.0 ~ 1.0
    gpu_id = 1

    data_dir = '/mnt/sdb1/data_sets'
    aerial_dir = os.path.join(data_dir, 'AerialImageDataset/train')
    cloud_dir = os.path.join(data_dir, 'ftp.nnvl.noaa.gov_color_IR_2018')
    grey_dir = os.path.join(data_dir, 'CloudGreyDataset_%dx%d' % (size, size))

    def __init__(self, model_dir='mod'):
        self.model_dir = model_dir
        self.model_name = 'mod'
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.model_npz = os.path.join(self.model_dir, self.model_name + '.npz')
        self.model_log = os.path.join(self.model_dir, 'training_npy.txt')


def run():
    import cv2
    import numpy as np

    idxs  = set([int(n[:6]) for n in os.listdir('result')])
    dilate_kernel = np.ones((3, 3))
    for i in idxs:
        aerial = cv2.imread('result/%06d-4-aerial.png' % i)
        mask01 = cv2.imread('result/%06d-0-cloud3.png' % i)
        mask01 = cv2.dilate(mask01, dilate_kernel)

        result = aerial
        cv2.imwrite('result_dict/%06d-3-result-thin.png' % i, result)

    pass






if __name__ == '__main__':
    # from mod_eval import run
    # from mod_replace import run

    # from mod_cloud_detect import run
    # from mod_cloud_remove import run

    run()
