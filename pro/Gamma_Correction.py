# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:25:14 2023

@author: cambridge
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

in_dir = "/home/akugyo/cv/in/"
in_fname = "dark_img1.jpg"
cur_iffname = in_dir + in_fname

out_dir = "/home/akugyo/cv/out/"
out_fname = "dark_img1_gamma_corrected.jpg"
cur_offname = out_dir + out_fname

in_img = cv2.imread(cur_iffname)
cv2.namedWindow("original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", 640,480)
plt.imshow(in_img)
plt.show()

cv2.namedWindow("Gamma corrected Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gamma corrected Image", 640,480)
out_img = np.array(255*(in_img/255)**0.45, dtype="uint8")
cv2.imwrite(cur_offname, out_img)
plt.imshow(out_img)
plt.show()