# -*- coding: utf-8 -*-
"""
Created by Dr.D.Antony Louis Piriyakumar

For 18AI742 Computer vision course

LOG 
"""

#importing the opencv module  
import cv2

img_in_dir = "D:\\Antony\\Lec\\CV\\Tutorials\\18AI00123\\In_data\\"
img_out_dir ="D:\\Antony\\Lec\\CV\\Tutorials\\18AI00123\\Out_data\\"

img_in_fileName = "Usain_Bolt1.jpg"
img_out_fileName = "O_Usain_Bolt1.jpg"

# img_in_fileName = "Chair1.jpg"
# img_out_fileName = "Chair1.jpg"

fu_img_in_fileName = img_in_dir + img_in_fileName
fu_img_out_fileName = img_out_dir + img_out_fileName

# using imread('path') and 0 denotes read as  grayscale image  
in_img = cv2.imread(fu_img_in_fileName,1)  

# [reduce_noise]
# Remove noise by blurring with a Gaussian filter
src = cv2.GaussianBlur(in_img, (3, 3), 0)
# [reduce_noise]

# Converting image to grayscale
gray= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
 
# Declare the variables we are going to use
ddepth = cv2.CV_16S
kernel_size = 3

# [laplacian]
# Apply Laplace function
dst = cv2.Laplacian(gray, ddepth, ksize=kernel_size)
# [laplacian]
# [convert]
# converting back to uint8
abs_dst = cv2.convertScaleAbs(dst)
# [convert]

disp_in_img_name = "Input Image"
#This is using for display the image  
cv2.imshow(disp_in_img_name,in_img)  

disp_out_img_name = "O LOG"
#This is using for display the image  
cv2.imshow(disp_out_img_name,abs_dst)  


cv2.waitKey(3) # This is necessary to be required so that the image doesn't close immediately.  
#It will run continuously until the key press.  
# cv2.destroyAllWindows()