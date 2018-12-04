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
    # Convert to LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Split image into L,A,B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    cl = clahe.apply(b_channel)

    # Merge channels
    equalised = cv2.merge( (l_channel, a_channel, cl) )

    # Convert back to BGR
    #final_image = cv2.cvtColor(equalised, cv2.COLOR_LAB2BGR)
    final_image = cv2.cvtColor(equalised, cv2.COLOR_HSV2BGR)
    return final_image


if __name__ == '__main__':

    # Command line argument handler
    parser = argparse.ArgumentParser(description="Example image operation over given image")
    parser.add_argument("-i","--image", help="Location of the image file from current directory")
    parser.add_argument("-d", "--directory", help="Directory of the image database")
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

    cv2.namedWindow('color')

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

    # cv2.createTrackbar('Overlay', 'color', 25, 100, nothing)
    while True:

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
        # overlay_weight = cv2.getTrackbarPos('Overlay', 'control')/100

        # Image preprocessing
        equalised = clahe_equalisation(img)
        # equalised = img
        cv2.imshow('Equalised', equalised)
        blurred = cv2.GaussianBlur(equalised, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Pplt HSV Values
        # plt.imshow(hsv)
        # plt.show()

        # Thresholding
        kernel = np.ones((9,9), np.uint8)

        mask1 = cv2.inRange(hsv, (H1_min, S1_min, V1_min), (H1_max, S1_max, V1_max))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

        mask2 = cv2.inRange(hsv, (H2_min, S2_min, V2_min), (H2_max, S2_max, V2_max))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

        mask3 = cv2.inRange(hsv, (H3_min, S3_min, V3_min), (H3_max, S3_max, V3_max))
        mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, kernel)
        mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, kernel)

        mask = mask1 + mask2 + mask3
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
            print("H1 Min: ", H1_min, "H1 Max: ", H1_max)
            print("S1 Min: ", S1_min, "S1 Max: ", S1_max)
            print("V1 Min: ", V1_min, "V1 Max: ", V1_max)

            print("H2 Min: ", H2_min, "H2 Max: ", H2_max)
            print("S2 Min: ", S2_min, "S2 Max: ", S2_max)
            print("V2 Min: ", V2_min, "V2 Max: ", V2_max)

            print("H3 Min: ", H3_min, "H3 Max: ", H3_max)
            print("S3 Min: ", S3_min, "S3 Max: ", S3_max)
            print("V3 Min: ", V3_min, "V3 Max: ", V3_max)
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
