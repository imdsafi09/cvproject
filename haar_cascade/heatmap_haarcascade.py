# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-11-02 10:12:00
# @Last Modified by:   vamshi
# @Last Modified time: 2018-12-14 07:54:18


# Script to take an input video. Process each frame to find the object using the cascade. Use heatmaps/maximum suppression method to combine the detections.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label


#find heatmaps of the bounding boxes
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    print bbox_list
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


#Draw the combined bounding box on the image
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected objects
    rects = []
    x1_list=[]
    y1_list=[]
    x2_list=[]
    y2_list=[]
    
    bbox_list=[]
    for object_number in range(1, labels[1]+1):
        # Find pixels with each object label value
        nonzero = (labels[0] == object_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)

        rects.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
  	
    return img, rects

#threshold the heatmap image 
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

#cascade object
toy_cascade= cv2.CascadeClassifier('cascade.xml')

#Benchmark video
cap = cv2.VideoCapture('Video-Lab.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frame_height=frame_height/2 #960
frame_width=frame_width/2 #540


# print frame_width,frame_height

out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

	
while(cap.isOpened()):
    ret, img = cap.read(0)

    print ret

    #Resizing the images
    img_viewing=cv2.resize(img,(frame_width,frame_height))
    gray = cv2.cvtColor(img_viewing, cv2.COLOR_BGR2GRAY)
    img_resized=cv2.resize(img,(192,108))
    gray_resized=cv2.resize(gray,(192,108)) 

    #detection object
    toys = toy_cascade.detectMultiScale(gray_resized, 1.2, 5)

    rectangles=[]

    #Rescaling the resized images with bounding box to original scale
    for (x,y,w,h) in toys:
        #Here frame_height/192=frame_width/108=5. Hence this is used to rescale the bounding box to the original image's scale
        x1=x*5 
        y1=y*5
        x2=(x+w)*5
        y2=(y+h)*5
        rectangles.append([(x1,y1),(x2,y2)])


    ########Using heatmaps to combine the detected bounding boxes#############
    heatmap_gray = np.zeros_like(img_viewing[:,:,0])
    heatmap_gray = add_heat(heatmap_gray, rectangles)
    # print heatmap_gray

    heatmap_gray =apply_threshold(heatmap_gray,0)

    labels = label(heatmap_gray)

    # Draw bounding boxes on a copy of the image
    img_viewing, rect = draw_labeled_bboxes(np.copy(img_viewing), labels)

    ###########################################################################

    cv2.imshow('img_viewing',img_viewing)
    # cv2.imshow('gray_resized',gray_resized)
    cv2.imshow('img_resized',img_resized)

    out.write(img_viewing)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()