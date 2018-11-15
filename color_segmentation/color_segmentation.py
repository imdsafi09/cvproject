#!/usr/bin/env python

import cv2
import numpy as np

import sys

# Empty function called when trackbar value is changed
def nothing(*arg):
    pass

if __name__ == '__main__':

    # Get location of the image from command line
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        img = cv2.resize(img, (0,0), fx=0.15, fy=0.15)
    else:
        print('usage : python color_segmentation <image file>')
        sys.exit()
    # Create window for displaying images
    cv2.namedWindow('color')

    # Create window for trackbar control
    cv2.namedWindow('control')
    cv2.createTrackbar('H1_min', 'control', 0, 180, nothing)
    cv2.createTrackbar('H1_max', 'control', 0, 180, nothing)

    cv2.createTrackbar('S1_min', 'control', 0, 255, nothing)
    cv2.createTrackbar('S1_max', 'control', 0, 255, nothing)

    cv2.createTrackbar('V1_min', 'control', 0, 255, nothing)
    cv2.createTrackbar('V1_max', 'control', 0, 255, nothing)

    #TODO:
    # Testing robustness: In the current window, have a key binding which loads the nect image and applies the same threshold values.
    # Objective: See what needs to be changed to detect the same color in different conditions.

    # cv2.createTrackbar('Overlay', 'color', 25, 100, nothing)
    while True:

        # Get trackbar positions
        H1_min = cv2.getTrackbarPos('H1_min', 'control')
        H1_max = cv2.getTrackbarPos('H1_max', 'control')

        S1_min = cv2.getTrackbarPos('S1_min', 'control')
        S1_max = cv2.getTrackbarPos('S1_max', 'control')

        V1_min = cv2.getTrackbarPos('V1_min', 'control')
        V1_max = cv2.getTrackbarPos('V1_max', 'control')

        overlay_weight = cv2.getTrackbarPos('Overlay', 'control')/100

        # Image preprocessing
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Thresholding
        kernel = np.ones((9,9), np.uint8)
        mask = cv2.inRange(hsv, (H1_min, S1_min, V1_min), (H1_max, S1_max, V1_max))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Mask visualisation
        # vis = img.copy()
        # vis = np.uint8(vis)
        # segment = cv2.bitwise_and(vis, np.uint8(mask))
        # vis = cv2.addWeighted(vis,(1-overlay_weight), segment, overlay_weight)

        cv2.imshow('color', img)
        cv2.imshow('threshold', mask)

        # Handling loop break
        ch = cv2.waitKey(1)
        if ch == 27:
            break

    # Close windows
    cv2.destroyAllWindows()
