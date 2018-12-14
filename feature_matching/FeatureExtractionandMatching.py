
# coding: utf-8

# In[ ]:


###############################################################################################################################
# Import definitions
from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt

###############################################################################################################################
# Import the template image to extract keypoints and descriptors
imgo = cv2.imread('Template.jpg',0)
imgs = cv2.imread('Brightness-2.jpg',0)
# Resizing the image (Mostly to display the image in the window)
img = cv2.resize(imgo,(0,0),fx = 0.2, fy = 0.2)
img1 = cv2.resize(imgs,(0,0),fx = 0.2, fy = 0.2)
# ############################################################################################################################
# Extra steps to improve the results for low brightness. Use the CLAHE equalization on both the template image and query image
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
#img = clahe.apply(imgre1)
#img1 = clahe.apply(imgre2)
################################################################################################################################
# SURF IMPLEMENTATION
# Set the Hessian Threshold. Changing this will change the number of keypoints detected. 
minHessian = 400
# Create a surf object
surf = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
# Extraxt keypoints and Descriptors for template and scene
keypoints_obj, descriptors_obj = surf.detectAndCompute(img, None)          # Template
keypoints_scene, descriptors_scene = surf.detectAndCompute(img1, None)     # Scene
################################################################################################################################
# FLANN based Matcher Implementation
# Create a matcher object (FLANN Based Matcher)
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
# Generate the KNN matches using the descriptors 
knn_matches = matcher.knnMatch(np.asarray(descriptors_obj,np.float32),np.asarray(descriptors_scene,np.float32), 2)
# Filter matches using the Lowe's ratio test
ratio_thresh = 0.8
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
# Draw matches
img_matches = np.empty((max(img.shape[0], img1.shape[0]), img.shape[1]+img1.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(img, keypoints_obj, img1, keypoints_scene, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# Localize the object
obj = np.empty((len(good_matches),2), dtype=np.float32)
scene = np.empty((len(good_matches),2), dtype=np.float32)
for i in range(len(good_matches)):
    # Get the keypoints from the good matches
    obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
    obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
    scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
    scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
# Homography with RANSAC for outlier rejection
H, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)
###############################################################################################################################
# Drawing the bounding box
# Get the corners from the image_1 (object to be detected)
obj_corners = np.empty((4,1,2), dtype=np.float32)
obj_corners[0,0,0] = 0
obj_corners[0,0,1] = 0
obj_corners[1,0,0] = img.shape[1]
obj_corners[1,0,1] = 0
obj_corners[2,0,0] = img.shape[1]
obj_corners[2,0,1] = img.shape[0]
obj_corners[3,0,0] = 0
obj_corners[3,0,1] = img.shape[0]
# Generate perspective transform
scene_corners = cv2.perspectiveTransform(obj_corners, H)
# Draw lines between the corners (the mapped object in the scene)
cv2.line(img_matches, (int(scene_corners[0,0,0] + img.shape[1]), int(scene_corners[0,0,1])),    (int(scene_corners[1,0,0] + img.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
cv2.line(img_matches, (int(scene_corners[1,0,0] + img.shape[1]), int(scene_corners[1,0,1])),    (int(scene_corners[2,0,0] + img.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
cv2.line(img_matches, (int(scene_corners[2,0,0] + img.shape[1]), int(scene_corners[2,0,1])),    (int(scene_corners[3,0,0] + img.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
cv2.line(img_matches, (int(scene_corners[3,0,0] + img.shape[1]), int(scene_corners[3,0,1])),    (int(scene_corners[0,0,0] + img.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
###############################################################################################################################
# Display results
# Show detected matches
cv2.imshow('Good Matches & Object detection', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

