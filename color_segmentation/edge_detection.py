#!/usr/bin/env python

import cv2
import numpy as np

import sys
import glob
import argparse

# Empty function called when trackbar value is changed
def nothing(*arg):
    pass

def readImage(location):
    img = cv2.imread(location)
    img = cv2.resize(img, (0,0), fx=0.15, fy=0.15)
    return img

if __name__ == '__main__':

    # Command line argument parser
    parser = argparse.ArgumentParser(description="Detect edges of the given image/set of images")
    parser.add_argument("-i", "--image", help="Location of the image file from current deirectory")
    parser.add_argument("-d", "--directory", help="Directory of the image database")
    args = parser.parse_args()

    # Get location of the image from command line
    if args.image:
        img = readImage(args.image)

    if args.directory:
        photos = glob.glob(args.directory + "*.jpg")
        counter = 0
        print(photos)
        img = readImage(photos[counter])

    # Create window for the image display
    cv2.namedWindow('edge')

    # Create trackbar for changing threshold
    cv2.namedWindow('control')
    cv2.createTrackbar('thrs1', 'control', 2000, 5000, nothing)
    cv2.createTrackbar('thrs2', 'control', 4000, 5000, nothing)
    cv2.createTrackbar('scale', 'control', 12, 50, nothing)

    while True:

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get trackbar positions
        thrs1 = cv2.getTrackbarPos('thrs1', 'control')
        thrs2 = cv2.getTrackbarPos('thrs2', 'control')
        scale = cv2.getTrackbarPos('scale', 'control')
        # Convert into odd number
        scale = scale+1 if scale%2 == 0 else scale

        # Scale change
        # print(scale)
        scaled = cv2.GaussianBlur(gray, (scale, scale), 0)

        # Canny edge detection
        edge = cv2.Canny(scaled, thrs1, thrs2, apertureSize=5)

        # Edge visualisation
        vis = img.copy()
        vis = np.uint8(vis/2.)
        vis[edge != 0] = (0, 255, 0)
        cv2.imshow('edge', vis)

        # Handling loop break
        ch = cv2.waitKey(5)
        if ch == 27:
            break

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

