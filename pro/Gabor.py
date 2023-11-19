# -*- coding: utf-8 -*-
"""
Created by Dr.D.Antony Louis Piriyakumar

For 18AI742 Computer vision course

Gabor
"""

#!/usr/bin/env python
## Gabor.py
## Author: Avi Kak (kak@purdue.edu)
## Date: October 15, 2016
## bugfix and changes on January 21, 2018:
##
## See the fix in Line (E7) that was needed to make the code work with the more recent
## Pillow library for PIL.
##
## Additionally, the code should now be Python 3 compliant.
## This script was written as a teaching aid for the lecture on "Textures and Color" as
## a part of my class on Computer Vision at Purdue.
##
## This Python script demonstrates how Gabor Filtering can be used for characterizing
## image textures.
## Just for the sake of playing with the code, this script generates five different types

## of "textures". You make the choice by uncommenting one of the statements in lines
## (A4) through (A8). You can also set the size of the image array and number of grayscale
## values to use in lines in lines (A9) and (A10)
## HOW TO USE THIS SCRIPT:
##
## 1. Specify the texture type you want by uncommenting one of the lines (A4) through (A8)
##
## 2. Set the image size in line (A9)
##
## 3. Set the number of gray levels in line (A10)
##
## 4. Set the size of the Gabor sigma in line (A29)
##
## 5. Set the size of the Gabor convolutional operator in line (A30).
## Call syntax: Gabor.py

import random
import math
import sys, glob, os

if sys.version_info[0] == 3:
    import tkinter as Tkinter
    from tkinter.constants import *
else:
    import Tkinter
    from Tkconstants import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageTk
import functools
import matplotlib.pyplot as plt

debug = True #(A1)
def main(): #(A2)
    if debug: #(A3)
        ## UNCOMMENT THE TEXTURE TYPE YOU WANT:
        # texture_type = ’random’ #(A4)
        texture_type = "vertical" #(A5)
        # texture_type = ’horizontal’ #(A6)
        # texture_type = ’checkerboard’ #(A7)
        # texture_type = None #(A8)
        IMAGE_SIZE = 32 #(A9)
        GRAY_LEVELS = 6 #(A10)
        image = [[0 for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)] #(A11)
        if texture_type == "random" : #(A12)
            image = [[random.randint(0,GRAY_LEVELS-1)
            for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)] #(A13)
        elif texture_type == "diagonal": #(A14)
            image = [[GRAY_LEVELS - 1 if (i+j)%2 == 0 else 0
            for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)] #(A15)
        elif texture_type == "vertical": #(A16)
            image = [[GRAY_LEVELS - 1 if i%2 == 0 else 0
            for i in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)] #(A17)
            plt.imshow(image)
            plt.show()
        elif texture_type == "horizontal": #(A18)
            image = [[GRAY_LEVELS - 1 if j%2 == 0 else 0
            for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)] #(A19)
        elif texture_type == "checkerboard": #(A20)
            image = [[GRAY_LEVELS - 1 if (i+j+1)%2 == 0 else 0
            for i in range(IMAGE_SIZE)] for j in range(IMAGE_SIZE)] #(A21)
        else: #(A22)
            sys.exit("You must satisfy a texture type by uncommenting one" +
                         "of lines (A1) through (A5).") #(A23)
        print("Texture type chosen: %s" % texture_type) #(A24)
        print("The image: ") #(A25)
        for row in range(IMAGE_SIZE): print(image[row]) #(A26)
    else: #(A27)
        sys.exit("Code for actual images goes here. Yet to be coded.") #(A28)
        

    gabor_sigma = 2.0 #(A29)
    gabor_size = 13 # must be odd #(A30)
    assert gabor_size % 2 == 1, "\n\nGabor filter size needs to be odd" #(A31)
    rowmin = colmin = gabor_size//2 #(A32)
    rowmax = colmax = IMAGE_SIZE - gabor_size//2 #(A33)
    how_many_frequencies = 2 #(A34)
    how_many_directions = 2 #(A35)
    gabor_filter_bank = generate_filter_bank(gabor_sigma, gabor_size,
    how_many_frequencies, how_many_directions) #(A36)
    directory_name = "filters" + str(gabor_size) #(A37)
    if os.path.isdir(directory_name): #(A38)
        list(map(os.remove, glob.glob(directory_name + "/*.jpg"))) #(A39)
    else: #(A40)
        os.mkdir(directory_name) #(A41)
    for item in glob.glob(directory_name + "/*"): os.remove(item) #(A42)
    filter_outputs = [[0.0 for _ in range(how_many_directions)]
                      for _ in range(how_many_frequencies)] #(A43)
    
    for freq in range(1,how_many_frequencies): #(A44)
        for direc in range(how_many_directions): #(A45)
            print("\n\nfilter for freq=%d and direction=%d:" % (freq,direc)) #(A46)
            print("\nop_real:") #(A47)
            op_real,op_imag = gabor_filter_bank[freq][direc] #(A48)
            display_gabor(op_real) #(A49)
            display_and_save_gabor_as_image(op_real,directory_name,
            "real_freq=%d_direc=%d"%(freq,direc)) #(A50)
            print("\nop_imag:") #(A51)
            display_gabor(op_imag) #(A52)
            display_and_save_gabor_as_image(op_imag,directory_name,
            "imag_freq=%d_direc=%d"%(freq,direc)) #(A53)
            for i in range(rowmin,rowmax): #(A54)
                for j in range(colmin,colmax): #(A55)
                    if debug: print("\n\nFor new pixel at (%d,%d):" % (i,j)) #(A56)
                    real_part,imag_part = 0.0,0.0 #(A57)
                    for k in range(-(gabor_size//2), gabor_size//2+1): #(A58)
                        for l in range(-(gabor_size//2), gabor_size//2+1): #(A59)
                            real_part += \
                            image[i-(gabor_size//2)+k][j-(gabor_size//2)+l] * \
                            op_real[-(gabor_size//2)+k][-(gabor_size//2)+l] #(A60)
                            imag_part += \
                            image[i-(gabor_size//2)+k][j-(gabor_size//2)+l] * \
                                op_imag[-(gabor_size//2)+k][-(gabor_size//2)+l] #(A61)
                    filter_outputs[freq][direc] += \
                    math.sqrt( real_part**2 + imag_part**2 ) #(A62)
    filter_outputs = list(map(lambda x: x / (1.0*(rowmax-rowmin)*(colmax-colmin)),
    filter_outputs[freq]) for freq in range(1,how_many_frequencies)) #(A63)
    print("\nGabor filter output:\n") #(A64)
    for freq in range(len(filter_outputs)): print(list(filter_outputs[freq])) #(A65)
    
    
def gabor(sigma, theta, f, size): #(B1)
    assert size >= 6 * sigma, \
    "\n\nThe size of the Gabor operator must be at least 6 times sigma" #(B2)
    W = size # Gabor operator Window width #(B3)
    coef = 1.0 / (math.sqrt(math.pi) * sigma) #(B4)
    ivals = range(-(W//2), W//2+1) #(B5)
    jvals = range(-(W//2), W//2+1) #(B6)
    greal = [[0.0 for _ in jvals] for _ in ivals] #(B7)
    gimag = [[0.0 for _ in jvals] for _ in ivals] #(B8)
    energy = 0.0 #(B9)
    for i in ivals: #(B10)
        for j in jvals: #(B11)
            greal[i][j] = coef * math.exp(-((i**2 + j**2)/(2.0*sigma**2))) *\
                math.cos(2*math.pi*(f/(1.0*W))*(i*math.cos(theta) +
                                                j * math.sin(theta))) #(B12)
            gimag[i][j] = coef * math.exp(-((i**2 + j**2)/(2.0*sigma**2))) *\
                math.sin(2*math.pi*(f/(1.0*W))*(i*math.cos(theta) +
                                                j * math.sin(theta))) #(B13)
            energy += greal[i][j] ** 2 + gimag[i][j] ** 2 #(B14)
    normalizer_r = functools.reduce(lambda x,y: x + sum(y), greal, 0) #(B15)
    normalizer_i = functools.reduce(lambda x,y: x + sum(y), gimag, 0) #(B16)
    print("\nnormalizer for the real part: %f" % normalizer_r) #(B17)
    print("normalizer for the imaginary part: %.10f" % normalizer_i) #(B18)
    print("energy: %f" % energy) #(B19)
    return (greal, gimag) #(B20)


def generate_filter_bank(sigma, size, how_many_frequencies,
                             how_many_directions): #(C1)
    filter_bank = {f : {d : None for d in range(how_many_directions)}
                   for f in range(how_many_frequencies)} #(C2)
    for freq in range(1,how_many_frequencies): #(C3)
        for direc in range(how_many_directions): #(C4)
            filter_bank[freq][direc] = \
                gabor(sigma, direc*math.pi/how_many_directions, 2*freq, size) #(C5)
    return filter_bank #(C6)

def display_gabor(oper): #(D1)
    height,width = len(oper), len(oper[0]) #(D2)
    for row in range(-(height//2), height//2+1): #(D3)
        sys.stdout.write("\n")
        for col in range(-(width//2), width//2+1): #(D4)
            sys.stdout.write("%5.2f" % oper[row][col])
    sys.stdout.write("\n")

def display_and_save_gabor_as_image(oper, directory_name, what_type): #(E1)
    height,width = len(oper), len(oper[0]) #(E2)
    maxVal = max(list(map(max, oper))) #(E3)
    minVal = min(list(map(min, oper))) #(E4)
    print("maxVal: %f" % maxVal) #(E5)
    print("minVal: %f" % minVal) #(E6)
    # newimage = Image.new("L", (width,height), 0.0) #(E7)
    newimage = Image.new("L", (width,height), 0) #(E7)
    for i in range(-(height//2), height//2+1): #(E8)
        for j in range(-(width//2),width//2+1): #(E9)
            if abs(maxVal-minVal) > 0: #(E10)
                displayVal = int((oper[i][j] - minVal) *
                                 (255/(maxVal-minVal))) #(E11)
            else: #(E12)
                displayVal = 0 #(E13)
            newimage.putpixel((j+width//2,i+height//2), displayVal) #(E14)
    displayImage3(newimage,directory_name, what_type, what_type +
                  " (close window when done viewing)") #(E15)
    
    
def displayImage3(argimage, directory_name, what_type, title=""): #(F1)
    """
    Displays the argument image in its actual size. The display stays on until the
    user closes the window. If you want a display that automatically shuts off after
    a certain number of seconds, use the method displayImage().
    """
    width,height = argimage.size #(F2)
    tk = Tkinter.Tk() #(F3)
    winsize_x,winsize_y = None,None #(F4)
    screen_width,screen_height = \
        tk.winfo_screenwidth(),tk.winfo_screenheight() #(F5)
    if screen_width <= screen_height: #(F6)
        winsize_x = int(0.5 * screen_width) #(F7)
        winsize_y = int(winsize_x * (height * 1.0 / width)) #(F8)
    else: #(F9)
        winsize_y = int(0.5 * screen_height) #(F10)
        winsize_x = int(winsize_y * (width * 1.0 / height)) #(F11)
    display_image = argimage.resize((winsize_x,winsize_y), Image.Resampling.LANCZOS) #(F12)
    image_name = directory_name + "/" + what_type #(F13)
    display_image.save(image_name + ".jpg") #(F14)
    tk.title(title) #(F15)
    frame = Tkinter.Frame(tk, relief=RIDGE, borderwidth=2) #(F16)
    frame.pack(fill=BOTH,expand=1) #(F17)
    photo_image = ImageTk.PhotoImage( display_image ) #(F18)
    label = Tkinter.Label(frame, image=photo_image) #(F19)
    label.pack(fill=X, expand=1) #(F20)
    tk.mainloop() #(F21)

main() #(G1)
