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
        img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    else:
        print('usage : python edge_detection <image file>')
        sys.exit()

    # Create trackbar for changing threshold
    cv2.namedWindow('edge')
    cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    cv2.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

    while True:

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get trackbar positions
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')

        # Canny edge detection
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)

        # Edge visualisation
        vis = img.copy()
        vis = np.uint8(vis/2.)
        vis[edge != 0] = (0, 255, 0)
        cv2.imshow('edge', vis)

        # Handling loop break
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    # Close windows
    cv2.destroyAllWindows()

