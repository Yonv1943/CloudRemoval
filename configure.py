import os


class Config(object):
    train_epoch = 2 ** 12
    train_size = 2 ** 15
    batch_size = min(2 ** 6, train_size)  # 2**13
    batch_epoch = train_size // batch_size

    test_size = 2 ** 2
    size = 2 ** 8  # 2 ** 6
    show_gap = 2 ** 2  # time
    save_gap = 2 ** 8  # time
    gpu_limit = 0.48  # 0.0 ~ 1.0
    learning_rate = 2 ** -10
    learning_beta = 2 ** -1
    max_gene_loss = 2 ** -3
    min_disc_loss = 2 ** -9
    epsilon = 2 ** -7

    model_dir = 'mod'
    model_name = 'mod'
    model_path = os.path.join(model_dir, model_name)
    model_npz = os.path.join(model_dir, model_name + '.npz')
    model_log = os.path.join(model_dir, 'training_npy.txt')

    data_dir = '/mnt/sdb1/data_sets'
    aerial_dir = os.path.join(data_dir, 'AerialImageDataset')
    cloud_dir = os.path.join(data_dir, 'ftp.nnvl.noaa.gov_color_IR_2018')
    grey_dir = os.path.join(data_dir, 'CloudGreyDataset')


if __name__ == '__main__':
    # from util.img_util import get_data_sets
    #
    # get_data_sets(4)

    from mod_mend import run

    run()
