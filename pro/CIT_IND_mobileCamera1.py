# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 23:05:00 2023

@author: cambridge
"""


# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
  
cur_dir_name = "/home/akugyo/cv/in/"
cur_fileName = "heya.jpg"
compl_FN = cur_dir_name + cur_fileName
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://100.73.250.75:8080/shot.jpg"
  #10.113.
# While loop to continuously fetching data from the Url
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    cv2.imshow("Android_cam", img)
  
    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break
    if(cv2.waitKey(10) & 0xFF == ord('x')):
        cv2.imwrite(compl_FN, img)
  
cv2.destroyAllWindows()


# Approach
# Download and install IP Webcam application on your mobile phone.
# Then make sure your PC and Phone both are connected to the same network. Open your IP Webcam 
# application on your both, click “Start Server” (usually found at the bottom). 
# This will open a camera on your Phone.
# A URL is being displayed on the Phone screen, type the same URL on your PC browser, and 
# under “Video renderer” Section, click on “Javascript”.
# You can see video captured on your phone, which starts showing up on your browser. 
# Now, what we will be going to do is, taking image data from the URL using 
# the request module and convert this to an image frame using NumPy, and finally, 
# start using our Android camera as a webcam in Python.
# In the code:
# Import module
# Add URL displayed in your phone
# Continuous fetch data from URL
# Keep displaying this data collected
# Close window
