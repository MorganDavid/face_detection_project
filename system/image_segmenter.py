#!/usr/bin/env python
'''
Usage:
	./ssearch.py input_image (f|q)
	f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''
 
import sys
import cv2
import time


class image_segmenter():
	def __init__(self): 
			print("made segmenter")
	def segment_image(self, im, method, numOfRectsToReturn):
	 
		# speed-up using multithreads
		cv2.setUseOptimized(True);
		cv2.setNumThreads(100);

		# resize image
		newHeight = 200
		newWidth = int(im.shape[1]*200/im.shape[0])
		im = cv2.resize(im, (newWidth, newHeight))	
		start = time.time()
		# create Selective Search Segmentation Object using default parameters
		ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	 
		# set input image on which we will run segmentation
		ss.setBaseImage(im)
	 
		# Switch to fast but low recall Selective Search method
		if (method == 'f'):
			ss.switchToSelectiveSearchFast()
	 
		# Switch to high recall but slow Selective Search method
		elif (method == 'q'):
			ss.switchToSelectiveSearchQuality()
		# if argument is neither f nor q print help message
		else:
			print(__doc__)
			sys.exit(1)
		end = time.time()
		print("time taken is ", end-start)
		# run selective search segmentation on input image
		rects = ss.process() # returns (x,y,w,h) for every segment
		
		retList = []
		# itereate over all the region proposals
		for i, rect in enumerate(rects):
			if(i < numOfRectsToReturn):
				x, y, w, h = rect
				retList.append((x,y,im[y:y+h,x:x+w]))

		return retList
if __name__ == "__main__":
   sgmtr = image_segmenter()
   rlist = sgmtr.segment_image(cv2.imread("1d_scaled.jpg"),"f",100)

