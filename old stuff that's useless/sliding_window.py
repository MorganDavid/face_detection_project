import cv2
import sys
import os
import cv2
import numpy as np
from cnn import makeModel,  MODEL_FILENAME
from keras.models import load_model
DETC_TRHESH = 1.5e-3#0.05#The threashold for it being a face (higher = more sensitive)

IM_WIDTH = 20
IM_HEIGHT = 20

#kernel_size is tupple: (height,width)
#Returns generator. Loop over this function
def make_sliding_window(image, step_size, kernel_size):
	
	image = cv2.resize(image,(new_width, new_height))
	x_up_to = new_width-kernel_size[0]-step_size
	y_up_to = new_height-kernel_size[1]-step_size
	for y in range(0,y_up_to,step_size):
		for x in range(0,x_up_to,step_size):
			yield (x,y,image[y:y+kernel_size[1],x:x+kernel_size[0]])
			
image = cv2.imread("1d_scaled.jpg")
height, width, depth = image.shape
print(image.shape)
resize_factor = 0.5
new_height, new_width = (int(height*resize_factor),int(width*resize_factor))
trumpim = cv2.resize(image,(new_width,new_height))
count = 0
images = []
positions = []
print("starting the sliding window loop. ")
for x,y,im in make_sliding_window(trumpim,2, (IM_HEIGHT,IM_WIDTH)):#Don't change kernel size, resize original image instead. Step size can be changed. 
	#cv2.imshow("t",im)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	print("looping through sliding window")
	images.append(im)
	positions.append((x,y))
	count = count + 1
	
model =load_model("test_model.h5")
images = np.asarray(images)
print(images.shape)

classes = model.predict(images, batch_size=10)
init_boxes = []; # will contain every box which has probability above the detection threashold. 
for i in range(0,len(classes)):
	this_class = classes[i]
	x,y = positions[i]
	print(i, '.jpg : ', this_class)
	if this_class<DETC_TRHESH:
		cv2.rectangle(trumpim, (x,y),(x+IM_WIDTH,y+IM_HEIGHT),(0,255,0),1) # draw all windows first. (no NMS)
		init_boxes.append([x,y,x+IM_WIDTH,y+IM_HEIGHT]);

# Malisiewicz et al.
# This function was taken from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

#perform NMS

supr_boxes = non_max_suppression(np.asarray(init_boxes),0.3);
for box in supr_boxes:
	cv2.rectangle(trumpim,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)

cv2.imshow("t",trumpim)
cv2.waitKey()
cv2.destroyAllWindows()
