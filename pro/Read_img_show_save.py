# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:00:09 2023

@author: cambridge
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:25:14 2023

@author: cambridge
"""

import cv2
import matplotlib.pyplot as plt

in_dir = "/home/akugyo/cv/in/"
in_fname = "dark_img1.jpg"
cur_iffname = in_dir + in_fname

out_dir = "/home/akugyo/cv/out/"
out_fname = "dark_img1_out.jpg"
cur_offname = out_dir + out_fname

in_img = cv2.imread(cur_iffname)
cv2.namedWindow("original Image", cv2.WINDOW_NORMAL)
plt.imshow(in_img)
plt.show()

out_img = in_img
cv2.imwrite(cur_offname, out_img)
plt.imshow(out_img)
plt.show()