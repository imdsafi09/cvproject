# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-11-30 00:27:30
# @Last Modified by:   vamshi
# @Last Modified time: 2018-12-04 17:40:30
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    # print bbox_list
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # # Assuming each "box" takes the form ((x1, y1), (x2, y2))
    
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    rects = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image and final rectangles
    return img, rects

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

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
    img_resized=cv2.resize(img,(192,108))
    gray_resized=cv2.resize(gray,(192,108))    
    img_viewing=cv2.resize(img,(view_h,view_w))
    toys = toy_cascade.detectMultiScale(gray_resized, 1.3, 15)

    rectangles=[]
    for (x,y,w,h) in toys:
        x1=x*4 
        y1=y*4
        x2=(x+w)*4
        y2=(y+h)*4
        rectangles.append([(x1,y1),(x2,y2)])
    	

    heatmap_gray = np.zeros_like(img_viewing[:,:,0])
    heatmap_gray = add_heat(heatmap_gray, rectangles)
    # print heatmap_gray

    heatmap_gray =apply_threshold(heatmap_gray,0)

    labels = label(heatmap_gray)

    print labels

    # Draw bounding boxes on a copy of the image
    draw_img, rect = draw_labeled_bboxes(np.copy(img_viewing), labels)
    # Display the image
    # plt.figure(figsize=(10,10))
    # plt.imshow(draw_img)


    # plt.show()
    cv2.imshow('draw_img',draw_img)
    cv2.imwrite('output_images/'+'output_'+i,draw_img)

    # cv2.waitKey(0)


    # for (x,y,w,h) in toys:
    #     cv2.rectangle(img_resized,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray_resized[y:y+h, x:x+w]
    #     roi_color = img_resized[y:y+h, x:x+w]
    
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# cv2.imshow('img_resized',img_resized)
# cv2.imshow('img_resized',img_resized)
# cv2.imshow('gray',gray)


# cv2.waitKey(0)

cv2.destroyAllWindows()