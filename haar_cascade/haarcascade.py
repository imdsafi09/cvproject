# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-11-02 10:12:00
# @Last Modified by:   vamshi
# @Last Modified time: 2018-12-13 21:02:19

#Script to take cascade as an input. Find the object in each frame of the input video. Write the output to a video


import numpy as np
import cv2


#cascade object
toy_cascade= cv2.CascadeClassifier('cascade.xml')

#Benchmark video
cap = cv2.VideoCapture('Video-Lab.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


#output video object
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

if (cap.isOpened()== False): 
  print("Error opening video stream or file")


# scale_match_x=frame_height/192
# scale_match_y=frame_width/108
while(cap.isOpened()):
    ret, img = cap.read(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Resizing the images
    img_resized=cv2.resize(img,(192,108))
    img_viewing=cv2.resize(img,(frame_width/2,frame_height/2))

    gray_resized=cv2.resize(gray,(192,108))    

    #detection object
    toys = toy_cascade.detectMultiScale(gray_resized, 1.2, 5 )

    # grouped_toys,weights=cv2.groupRectangles(list(toys), 3,0)

    # print grouped_toys,weights
    # print toys

    #Rescaling the resized images with bounding box to original scale
    for (x,y,w,h) in toys:
        cv2.rectangle(img_resized,(x,y),(x+w,y+h),(255,0,0),4)
        # scale_match_x=frame_height/192 #5
        # scale_match_y=frame_width/108 #5

        #Here frame_height/192=frame_width/108=5. Hence this is used to rescale the bounding box to the original image's scale
        x1=x*5 
        y1=y*5
        x2=(x+w)*5
        y2=(y+h)*5
        cv2.rectangle(img_viewing,(x1,y1),(x2,y2),(255,0,0),12)


    cv2.imshow('img_viewing',img_viewing)
    # cv2.imshow('img_resized',img_resized)

    out.write(img_viewing)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()