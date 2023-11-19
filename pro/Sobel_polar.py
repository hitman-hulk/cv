# -*- coding: utf-8 -*-
"""
Created by Dr.D.Antony Louis Piriyakumar

For 18AI742 Computer vision course

Sobel + polar
"""

#importing the opencv module  
import cv2  
import numpy as np
import matplotlib.pyplot as plt

img_in_dir = "/home/akugyo/Documents/GitHub/cv/in/"
img_out_dir ="/home/akugyo/Documents/GitHub/cv/out/"

img_in_fileName = "heh.jpg"
img_out_fileName = "O_Usain_Bolt1.jpg"

# img_in_fileName = "Chair1.jpg"
# img_out_fileName = "Chair1.jpg"

fu_img_in_fileName = img_in_dir + img_in_fileName
fu_img_out_fileName = img_out_dir + img_out_fileName

# using imread('path') and 0 denotes read as  grayscale image  
in_img = cv2.imread(fu_img_in_fileName,1)  

 # Converting image to grayscale
gray= cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
 
img = np.float32(in_img) / 255.0
 
# Calculate gradient
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# Python Calculate gradient magnitude and direction ( in degrees )
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)


disp_in_img_name = "Input Image"
#This is using for display the image  
plt.imshow(in_img)  
plt.show()

disp_out_img_name = "Output Image X"
#This is using for display the image  
plt.imshow(gx)  
plt.show()

disp_out_img_name = "Output Image Y"
#This is using for display the image  
plt.imshow(gy)  
plt.show()

disp_out_img_name = "Output Image Mag"
#This is using for display the image  
plt.imshow(mag)  
plt.show()

disp_out_img_name = "Output Image Angle"
#This is using for display the image  
plt.imshow(angle) 
plt.show()


cv2.waitKey(3) # This is necessary to be required so that the image doesn't close immediately.  
#It will run continuously until the key press.  
# cv2.destroyAllWindows()