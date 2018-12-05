#!/usr/bin/env python

import cv2
import numpy as np

import sys
import argparse
import glob
from matplotlib import pyplot as plt
import re

# Empty function called when trackbar value is changed
def nothing(*arg):
    pass

# Read the image by resizing the image a proper amount
def readImage(location):
    img = cv2.imread(location)
    img = cv2.resize(img, (0,0), fx=0.15, fy=0.15)
    return img

# Functions for natural sort
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


# Equalise image using CLAHE
def clahe_equalisation(img):
    # Convert to HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Split image into L,A,B channels
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    cl = clahe.apply(v_channel)

    # Merge channels
    equalised = cv2.merge( (h_channel, s_channel, cl) )

    # Convert back to BGR
    final_image = cv2.cvtColor(equalised, cv2.COLOR_HSV2BGR)
    return final_image

def normal_equalisation(img):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    cl = cv2.equalizeHist(v_channel)

    # Merge channels
    equalised = cv2.merge( (h_channel, s_channel, cl) )

    # Convert back to BGR
    final_image = cv2.cvtColor(equalised, cv2.COLOR_HSV2BGR)
    return final_image

def Trackbars():
    # Create window for trackbar control
    cv2.namedWindow('body')
    cv2.createTrackbar('H1_min', 'body', 27, 180, nothing)
    cv2.createTrackbar('H1_max', 'body', 79, 180, nothing)

    cv2.createTrackbar('S1_min', 'body', 33, 255, nothing)
    cv2.createTrackbar('S1_max', 'body', 202, 255, nothing)

    cv2.createTrackbar('V1_min', 'body', 0, 255, nothing)
    cv2.createTrackbar('V1_max', 'body', 208, 255, nothing)

    cv2.namedWindow('beard')
    cv2.createTrackbar('H2_min', 'beard', 0, 180, nothing)
    cv2.createTrackbar('H2_max', 'beard', 50, 180, nothing)

    cv2.createTrackbar('S2_min', 'beard', 147, 255, nothing)
    cv2.createTrackbar('S2_max', 'beard', 249, 255, nothing)

    cv2.createTrackbar('V2_min', 'beard', 135, 255, nothing)
    cv2.createTrackbar('V2_max', 'beard', 251, 255, nothing)

    cv2.namedWindow('face')
    cv2.createTrackbar('H3_min', 'face', 0, 180, nothing)
    cv2.createTrackbar('H3_max', 'face', 19, 180, nothing)

    cv2.createTrackbar('S3_min', 'face', 0, 255, nothing)
    cv2.createTrackbar('S3_max', 'face', 100, 255, nothing)

    cv2.createTrackbar('V3_min', 'face', 146, 255, nothing)
    cv2.createTrackbar('V3_max', 'face', 255, 255, nothing)

class Trackbar():
    def __init__(self, name, H_default=(0,0), S_default=(0,0), V_default=(0,0)):
        # Name of the trackbar window
        self.name = str(name)
        # Define default value of the trackbars
        self.H_min = H_default[0]
        self.H_max = H_default[1]
        self.S_min = S_default[0]
        self.S_max = S_default[1]
        self.V_min = V_default[0]
        self.V_max = V_default[1]

    def show_trackbars(self):
        # Create window for trackbar control
        cv2.namedWindow(self.name)
        cv2.createTrackbar('H_min', self.name, self.H_min, 180, nothing)
        cv2.createTrackbar('H_max', self.name, self.H_max, 180, nothing)

        cv2.createTrackbar('S_min', self.name, self.S_min, 255, nothing)
        cv2.createTrackbar('S_max', self.name, self.S_max, 255, nothing)

        cv2.createTrackbar('V_min', self.name, self.V_min, 255, nothing)
        cv2.createTrackbar('V_max', self.name, self.V_max, 255, nothing)

    def update_trackbar_values(self):
        # Get trackbar positions
        self.H_min = cv2.getTrackbarPos('H_min', self.name)
        self.H_max = cv2.getTrackbarPos('H_max', self.name)

        self.S_min = cv2.getTrackbarPos('S_min', self.name)
        self.S_max = cv2.getTrackbarPos('S_max', self.name)

        self.V_min = cv2.getTrackbarPos('V_min', self.name)
        self.V_max = cv2.getTrackbarPos('V_max', self.name)

def get_track():
    # Get trackbar positions
    H1_min = cv2.getTrackbarPos('H1_min', 'body')
    H1_max = cv2.getTrackbarPos('H1_max', 'body')

    S1_min = cv2.getTrackbarPos('S1_min', 'body')
    S1_max = cv2.getTrackbarPos('S1_max', 'body')

    V1_min = cv2.getTrackbarPos('V1_min', 'body')
    V1_max = cv2.getTrackbarPos('V1_max', 'body')

    H2_min = cv2.getTrackbarPos('H2_min', 'beard')
    H2_max = cv2.getTrackbarPos('H2_max', 'beard')

    S2_min = cv2.getTrackbarPos('S2_min', 'beard')
    S2_max = cv2.getTrackbarPos('S2_max', 'beard')

    V2_min = cv2.getTrackbarPos('V2_min', 'beard')
    V2_max = cv2.getTrackbarPos('V2_max', 'beard')

    H3_min = cv2.getTrackbarPos('H3_min', 'face')
    H3_max = cv2.getTrackbarPos('H3_max', 'face')

    S3_min = cv2.getTrackbarPos('S3_min', 'face')
    S3_max = cv2.getTrackbarPos('S3_max', 'face')

    V3_min = cv2.getTrackbarPos('V3_min', 'face')
    V3_max = cv2.getTrackbarPos('V3_max', 'face')



if __name__ == '__main__':

    # Command line argument handler
    parser = argparse.ArgumentParser(description="Example image operation over given image")
    parser.add_argument("-i","--image", help="Location of the image file from current directory")
    parser.add_argument("-d", "--directory", help="Directory of the image database")
    parser.add_argument("-v", "--video", help="Location of the video")
    args = parser.parse_args()

    # Get location of the image from command line
    if args.image:
        img = readImage(args.image)

    if args.directory:
        photos = glob.glob(args.directory + "*.jpg")
        photos.extend(glob.glob(args.directory + "*.png"))
        counter = 0
        print(photos)
        img = readImage(photos[counter])

    if args.video:
        print("Hellp, world!")
        quit()

    cv2.namedWindow('color')


    face = Trackbar('face'  , (0, 19) , (0, 100)  , (146, 255))
    beard = Trackbar('beard', (0, 50) , (147, 249), (135, 251))
    body = Trackbar('body'  , (27, 79), (33, 202) , (0, 208))

    face.show_trackbars()
    beard.show_trackbars()
    body.show_trackbars()

    # cv2.createTrackbar('Overlay', 'color', 25, 100, nothing)
    while True:

        # Get current trackbar values
        face.update_trackbar_values()
        beard.update_trackbar_values()
        body.update_trackbar_values()

        # Image preprocessing
        equalised = clahe_equalisation(img)
        # equalised = img
        cv2.imshow('Equalised', equalised)
        blurred = cv2.GaussianBlur(equalised, (21, 21), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Thresholding
        kernel = np.ones((9,9), np.uint8)

        mask1 = cv2.inRange(hsv, (face.H_min, face.S_min, face.V_min), (face.H_max, face.S_max, face.V_max))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

        mask2 = cv2.inRange(hsv, (beard.H_min, beard.S_min, beard.V_min), (beard.H_max, beard.S_max, beard.V_max))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

        mask3 = cv2.inRange(hsv, (body.H_min, body.S_min, body.V_min), (body.H_max, body.S_max, body.V_max))
        mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel)
        mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)

        mask = mask1 + mask2 + mask3
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Mask visualisation
        vis = img.copy()
        vis = np.uint8(vis)
        segment = cv2.bitwise_and(vis, vis, mask=mask)
        overlay_weight = 0
        vis = cv2.addWeighted(vis,(overlay_weight), segment, 1-overlay_weight, 0)

        cv2.imshow('color', img)
        cv2.imshow('threshold', vis)

        # Handling loop break
        ch = cv2.waitKey(1)
        if ch == 27:
            break
        # Write the images
        if ch == 119:
            result_dir = "results/"
            cur_images = glob.glob(result_dir + "*.jpg")
            cur_images.sort(key=natural_keys)

            if cur_images:
                latest_image = cur_images[-1]
                latest_image = latest_image.replace('.','_').split('_')
                count = int(latest_image[-2])+1
            else:
                count = 1

            # Write current images
            print("Writing the images")
            print(count)
            cv2.imwrite(result_dir+'figure_'+'original_'+str(count)+'.jpg', img)
            cv2.imwrite(result_dir+'figure_'+'equalised_'+str(count)+'.jpg', equalised)
            cv2.imwrite(result_dir+'figure_'+'mask_'+str(count)+'.jpg', vis)
            print("Finished writing.")

        # Print current values
        if ch == 113:
            print(face.name + " values: ")
            print("H Min: ", face.H_min, "H Max: ", face.H_max)
            print("S Min: ", face.S_min, "S Max: ", face.S_max)
            print("V Min: ", face.V_min, "V Max: ", face.V_max)

            print(beard.name + " values: ")
            print("H Min: ", beard.H_min, "H Max: ", beard.H_max)
            print("S Min: ", beard.S_min, "S Max: ", beard.S_max)
            print("V Min: ", beard.V_min, "V Max: ", beard.V_max)

            print(body.name + " values: ")
            print("H Min: ", body.H_min, "H Max: ", body.H_max)
            print("S Min: ", body.S_min, "S Max: ", body.S_max)
            print("V Min: ", body.V_min, "V Max: ", body.V_max)
        # Changing the image
        if args.directory:
            if photos:
                if ch == 106:
                    counter = (counter + 1) % len(photos)
                    img = readImage(photos[counter])
                elif ch == 108:
                    counter = (counter - 1) % len(photos)
                    img = readImage(photos[counter])


    # Close windows
    cv2.destroyAllWindows()
