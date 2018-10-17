import os


class Config(object):
    train_epoch = 2 ** 9
    train_size = 2 ** 15  # 15
    batch_size = min(2 ** 5, train_size)  # 2**13
    batch_epoch = train_size // batch_size

    test_size = 2 ** 2
    size = 2 ** 8  # 2 ** 6
    show_gap = 2 ** 6  # time
    save_gap = 2 ** 10  # time
    gpu_limit = 0.48  # 0.0 ~ 1.0

    data_dir = '/mnt/sdb1/data_sets'
    aerial_dir = os.path.join(data_dir, 'AerialImageDataset')
    cloud_dir = os.path.join(data_dir, 'ftp.nnvl.noaa.gov_color_IR_2018')
    grey_dir = os.path.join(data_dir, 'CloudGreyDataset')

    def __init__(self, model_dir='mod'):
        self.model_dir = model_dir
        self.model_name = 'mod'
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.model_npz = os.path.join(self.model_dir, self.model_name + '.npz')
        self.model_log = os.path.join(self.model_dir, 'training_npy.txt')


if __name__ == '__main__':
    from mod_WGAN import run
    # from mod_mend import run

    run()
