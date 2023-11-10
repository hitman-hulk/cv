# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 22:25:24 2023

@author: cambridge
"""

# import the opencv library
import cv2

cur_dir_name = "/home/akugyo/cv/in/"
cur_fileName = "my_fig.jpg"
compl_FN = cur_dir_name + cur_fileName
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
# ret,img=cap.read()
# cv2.imwrite(compl_FN, img) 
while True: 
    
    ret,img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Video', img)
    
    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break
    if(cv2.waitKey(10) & 0xFF == ord('x')):
        cv2.imwrite(compl_FN, img)
# Using cv2.imwrite() method
# Saving the image
#cv2.imwrite(compl_FN, img)

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()