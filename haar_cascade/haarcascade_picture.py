# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-11-30 00:27:30
# @Last Modified by:   vamshi
# @Last Modified time: 2018-12-04 17:29:40
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

toy_cascade= cv2.CascadeClassifier('cascade.xml')

# img=cv2.imread('Standard.jpg',1)

files=os.listdir('test_images')
print files

for i in files:
    img=cv2.imread('test_images/'+i,1)
    # img=cv2.imread('Ocluded-3.jpg',1)

    (w,h) = img.shape[:2] 
    view_h=h/6
    view_w=w/6

    print view_w/108
    print view_h/192

    # M = cv2.getRotationMatrix2D(center, 90, 1.0)  
    # img = cv2.warpAffine(img, M, (h, w)) 


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_viewing=cv2.resize(img,(view_h,view_w))
    img_resized=cv2.resize(img,(192,108))
    gray_resized=cv2.resize(gray,(192,108))    
    toys = toy_cascade.detectMultiScale(gray_resized, 1.3, 15)



    for (x,y,w,h) in toys:
        x1=x*4
        y1=y*4
        x2=(x+w)*4
        y2=(y+h)*4

        cv2.rectangle(img_viewing,(x1,y1),(x2,y2),(255,0,0),2)

        

    # cv2.imshow('img_resized',img_resized)
    # cv2.imshow('img_resized',img_resized)
    # cv2.imshow('gray',gray)
    cv2.imshow('img_viewing',img_viewing)
    cv2.imwrite('output_images/'+'output_'+i,img_viewing)

    cv2.waitKey(0)

    cv2.destroyAllWindows()