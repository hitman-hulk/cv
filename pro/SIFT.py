# -*- coding: utf-8 -*-
"""
Created by Dr.D.Antony Louis Piriyakumar

For 18AI742 Computer vision course

SIFT features
"""

#importing the opencv module  
import cv2  

img_in_dir = "/home/akugyo/Documents/GitHub/cv/in/"
img_out_dir ="/home/akugyo/Documents/GitHub/cv/out/"

img_in_fileName = "heh.jpg"
img_out_fileName = "O_Koeln_Kirche.jpg"

# img_in_fileName = "Chair1.jpg"
# img_out_fileName = "Chair1.jpg"

fu_img_in_fileName = img_in_dir + img_in_fileName
fu_img_out_fileName = img_out_dir + img_out_fileName

# using imread('path') and 0 denotes read as  grayscale image  
in_img = cv2.imread(fu_img_in_fileName,1)  
rows, cols = in_img.shape[:2]
img_shrinked = cv2.resize(in_img, (250, 200),
                         interpolation=cv2.INTER_AREA)
img_rotation = cv2.warpAffine(img_shrinked,
                             cv2.getRotationMatrix2D((cols/2, rows/2),
                                                    30, 0.6),
                             (cols, rows))

 # Converting image to grayscale
gray= cv2.cvtColor(img_rotation,cv2.COLOR_BGR2GRAY)
 
# Applying SIFT detector
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
 
# Marking the keypoint on the image using circles
out_img=cv2.drawKeypoints(gray ,
                      kp ,
                      in_img,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite(fu_img_out_fileName,out_img) 

disp_in_img_name = "Input Image"
#This is using for display the image  
cv2.imshow(disp_in_img_name,in_img)  

disp_out_img_name = "Output Image"
#This is using for display the image  
cv2.imshow(disp_out_img_name,out_img)  

cv2.waitKey(3) # This is necessary to be required so that the image doesn't close immediately.  
#It will run continuously until the key press.  
# cv2.destroyAllWindows()

