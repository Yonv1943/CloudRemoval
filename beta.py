import cv2
import numpy as np

"""
2018-11-05 何凯明 http://kaiminghe.com/cvpr09/
2018-11-05 Yonv1943, Reproduction in opencv-python 
"""


cv2.namedWindow('beta', cv2.WINDOW_KEEPRATIO)

img = cv2.imread('tiananmen1.png')
cv2.imshow('beta', img)
cv2.waitKey(1234)
