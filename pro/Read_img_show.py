# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:57:29 2023

@author: cambridge
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:25:14 2023

@author: cambridge
"""

import cv2

in_dir = "/home/akugyo/cv/in/"
in_fname = "dark_img1.jpg"
cur_iffname = in_dir + in_fname

in_img = cv2.imread(cur_iffname)
cv2.namedWindow("original Image", cv2.WINDOW_NORMAL)
cv2.imshow("original Image",in_img)

cv2.waitKey(0)