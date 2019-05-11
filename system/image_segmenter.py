#!/usr/bin/env python

import sys
import cv2
import time

if __name__ == '__main__':
    # If image path and f/q is not passed as command
    # line arguments, quit and display help message
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
 
    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);
 	
    # read image
    im = cv2.imread(sys.argv[1])
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
    if (sys.argv[2] == 'f'):
        ss.switchToSelectiveSearchFast()
 
    # Switch to high recall but slow Selective Search method
    elif (sys.argv[2] == 'q'):
        ss.switchToSelectiveSearchQuality()
    # if argument is neither f nor q print help message
    else:
        print(__doc__)
        sys.exit(1)
 
    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))
    end = time.time()
    print(end - start)

    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 20
    
    while True:
        # create a copy of original image
        imOut = im.copy()
 
        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break
   
        # show output
        cv2.imshow("Output", imOut)
 
        # record key press
        k = cv2.waitKey(0) & 0xFF
 
        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
            print(numShowRects);
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
            print(numShowRects);
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()

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