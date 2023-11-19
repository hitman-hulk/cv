# -*- coding: utf-8 -*-
"""
Created by Dr.D.Antony Louis Piriyakumar

For 18AI742 Computer vision course

LOG 
"""

#importing the opencv module  
import cv2
import numpy as np

img_in_dir = "D:\\Antony\\Lec\\CV\\Tutorials\\18AI00123\\In_data\\"
img_out_dir ="D:\\Antony\\Lec\\CV\\Tutorials\\18AI00123\\Out_data\\"

img_in_fileName = "Texture2.jpg"
img_out_fileName = "O_Texture2.jpg"

# img_in_fileName = "Chair1.jpg"
# img_out_fileName = "Chair1.jpg"

fu_img_in_fileName = img_in_dir + img_in_fileName
fu_img_out_fileName = img_out_dir + img_out_fileName

# using imread('path') and 0 denotes read as  grayscale image  
in_img = cv2.imread(fu_img_in_fileName,0)  

E5 = np.array([[-1,2,0,2,1]])
L5 = np.array([[1,4,6,4,1]])

kernel_E5L5 = np.multiply(E5.transpose(),L5)
# Creating the kernel(2d convolution matrix) 

# Applying the filter2D() function 
out_img = cv2.filter2D(src=in_img, ddepth=-1, kernel=kernel_E5L5) 

cv2.imwrite(fu_img_out_fileName,out_img) 

disp_in_img_name = "Input Image"
#This is using for display the image  
cv2.imshow(disp_in_img_name,in_img)  

disp_out_img_name = "O E5L5"
#This is using for display the image  
cv2.imshow(disp_out_img_name,out_img)  

cv2.waitKey(3) # This is necessary to be required so that the image doesn't close immediately.  
#It will run continuously until the key press.  
# cv2.destroyAllWindows()
