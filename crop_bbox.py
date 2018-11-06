#Mahdi Elhousni, 06/11/2018, WPI.

import numpy as np
import cv2
import re

#crop images inside of the pre-defined boundingbox


for i in range(0,216):

	ref = str(i)
	path = '/media/mahdielh/usb/0/' + ref
	path2 = '/media/mahdielh/usb/1/' + ref

	#print ref

	with open(path+'.xml', 'r') as inF:


    		for line in inF:

        		if '<xmin>' in line:
	    		    	result = re.search('<xmin>(.*)</xmin>', line)
	    			xmin = int(result.group(1))
				#print xmin

        		if '<ymin>' in line:
	    			result = re.search('<ymin>(.*)</ymin>', line)
	    			ymin = int(result.group(1))
				#print ymin

        		if '<xmax>' in line:
	    			result = re.search('<xmax>(.*)</xmax>', line)
	    			xmax = int(result.group(1))
				#print xmax

        		if '<ymax>' in line:
	    			result = re.search('<ymax>(.*)</ymax>', line)
	    			ymax = int(result.group(1))
				#print ymax

	test=path+'.png'
	print test

	im = cv2.imread(path+'.png')
	print ymax
	##letter = im[y:y+h,x:x+w]
	toy = im[ymin:ymax,xmin:xmax]
	cv2.imwrite(path2 +'-cropped.png', toy)


