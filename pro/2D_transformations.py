# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:11:49 2023

@author: cambridge
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


in_dir = "/home/akugyo/cv/in/"
in_fname = "dhoni-dive"
cur_iffname = in_dir + in_fname

out_dir = "/home/akugyo/cv/out/"
out_fname = "Dhoni-dive_ifft.jpg"
cur_offname = out_dir + out_fname

img = cv2.imread(cur_iffname,0)
rows, cols = img.shape
translation_x = 100
translation_y = 50
trans_M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
dst = cv2.warpAffine(img, trans_M, (cols, rows))
plt.imshow(dst)
plt.show()

reflec_MX = np.float32([[1,  0, 0],
                [0, -1, rows],
                [0,  0, 1]])
reflected_img = cv2.warpPerspective(img, reflec_MX,
                                   (int(cols),
                                    int(rows)))
plt.imshow(reflected_img)
plt.show()
img_rotation = cv2.warpAffine(img,
                             cv2.getRotationMatrix2D((cols/2, rows/2),
                                                    30, 0.6),
                             (cols, rows))
plt.imshow(img)
plt.show()
plt.imshow(img_rotation)
plt.show()
img_shrinked = cv2.resize(img, (250, 200),
                         interpolation=cv2.INTER_AREA)
plt.imshow(img_shrinked)
plt.show()
img_enlarged = cv2.resize(img_shrinked, None,
                         fx=1.5, fy=1.5,
                         interpolation=cv2.INTER_CUBIC)
plt.imshow(img_enlarged)
plt.show()
shear_MX = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
sheared_img = cv2.warpPerspective(img, shear_MX, (int(cols*1.5), int(rows*1.5)))
plt.imshow(sheared_img)
plt.show()