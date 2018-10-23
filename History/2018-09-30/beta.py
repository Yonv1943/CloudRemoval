import cv2
import numpy as np
import os

os_join = os.path.join

from configure import Config

C = Config


def auto_canny(img, sigma=0.5):
    v = np.median(img)
    lower = int(max((1.0 - sigma) * v, 0))
    upper = int(max((1.0 + sigma) * v, 255))

    return cv2.Canny(img, lower, upper)


img_path = os_join(C.data_dir, 'austin27.tif')
img = cv2.imread(img_path)  # shape == (5000, 5000, 3)
img = img[-900:, -900:]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.Laplacian(img, ddepth=cv2.CV_8U, ksize=3)

cv2.namedWindow('beta', cv2.WINDOW_KEEPRATIO)
cv2.imshow('beta', img)
cv2.waitKey(3456)
