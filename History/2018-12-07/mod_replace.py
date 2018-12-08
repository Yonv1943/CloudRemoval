import cv2
import numpy as np

from configure import Config
from util import img_util

"""
Enter aerial images taken at different times in the same place,
and select the thinest area of the cloud to replace the original image.
2018-06-30 Source: Yonv1943
2018-10-30 timeline replace
"""

C = Config

cv2.namedWindow('beta', cv2.WINDOW_KEEPRATIO)


# cv2.imshow('beta', img)
# cv2.waitKey(1234)


def run():  # timeline_replace
    ground_path = '/mnt/sdb1/data_sets/AerialImageDataset/test/tyrol-e24.tif'
    ground = cv2.imread(ground_path)  # shape == (5000, 5000, 3)
    ground = ground[3000:4060, 3000:4920]

    cloud1s = img_util.get_cloud1_continusly(1943, 1943 + 128, 4)
    aerials = img_util.get_aerial_continusly(ground, cloud1s)

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

    return out_aer, out_cld, ground, cloud1s, aerials


if __name__ == '__main__':
    # from mod_replace import timeline_replace
    out_aer, out_cld, ground, cloud1s, aerials = run()

