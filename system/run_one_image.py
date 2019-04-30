import cv2
import sys
import os
import cv2
import numpy as np
from keras.models import load_model
import time

class run_one_image():
	DETC_TRHESH = -0.99999999#The threashold for it being a face (higher = more sensitive) This is used when we need to detect more than one face in the image.harry: 1.5e11 1d: 1.5e4
	NUM_FACES = 80 # This is used instead of DETEC_THREASH. Use the top_k_faces function instead. NUM_FACES=2 means take the top 2 most confident face boxes and draw them.
	IM_WIDTH = 12 # this actually the dims the CNN was trained on. 
	IM_HEIGHT = 12
	_model_dir = r'multi-output.h5'

	def __init__(self):
		print("made class")
	'''
	kernel_size is tupple: (height,width)
	Returns generator. Loop over this function
	'''
	def make_sliding_window(self, image, step_size, kernel_size):
		height, width, _ = image.shape
		image = cv2.resize(image,(width, height))
		x_up_to = width-kernel_size[0]-step_size
		y_up_to = height-kernel_size[1]-step_size
		for y in range(0,y_up_to,step_size):
			for x in range(0,x_up_to,step_size):
				yield (x,y,image[y:y+kernel_size[1],x:x+kernel_size[0]])
				
	
	# Malisiewicz et al.
	# This function was taken from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	def non_max_suppression(self, boxes, overlapThresh):
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
	'''
	# This bit takes every box above the confidence threash and performs NMS.
	supr_boxes = non_max_suppression(np.asarray(init_boxes),0.3);
	for box in supr_boxes:
		cv2.rectangle(trumpim,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
	'''
	def draw_from_threashold(self, img, init_boxes):
		supr_boxes = self.non_max_suppression(np.asarray(init_boxes),0.3);
		for box in supr_boxes:
			cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
	
	#This method takes the top K confidence boxes and uses them. Also performs NMS.
	def draw_top_k_rectangles(self, img, classes, positions,k):
		indices = np.argpartition(classes.flatten(),k)[:k]
		boxes_top_k_xy = [positions[i] for i in indices] # get x,y of every box
		boxes_top_k = [[i[0],i[1],i[0]+self.IM_WIDTH,i[1]+self.IM_HEIGHT] for i in boxes_top_k_xy] # make them four coords. 
		print(boxes_top_k)
		boxes_nms = self.non_max_suppression(np.asarray(boxes_top_k),0.28)
		for box in boxes_nms:
			x,y=box[:2]
			cv2.rectangle(img,(x,y),(x+self.IM_WIDTH,y+self.IM_HEIGHT),(0,255,0),1)
	#draw_top_k_rectangles(100)

	'''
	Runner method '''
	def detect_faces(self, image):
		height, width, depth = image.shape
		print(image.shape)
		resize_factor = 0.2
		new_height, new_width = (int(height*resize_factor),int(width*resize_factor))
		trumpim = cv2.resize(image,(new_width,new_height))
		count = 0
		images = []
		positions = []
		print("starting the sliding window loop. ")
		start = time.time()
		for x,y,im in self.make_sliding_window(trumpim,2, (self.IM_HEIGHT,self.IM_WIDTH)):#Don't change kernel size, resize original image instead. Step size can be changed. 
			images.append(im)
			positions.append((x,y))
			count = count + 1

		print("sliding window took ",time.time()-start, " seconds.")	
		model =load_model(self._model_dir)
		images = np.asarray(images)

		print("starting prediction")
		start = time.time()
		classes = model.predict(images, batch_size=50)
		print("prediction took ",time.time()-start, " seconds.")
		binary = [x-y for (x,y) in classes[0]]
		bboxes = classes[1]
		np.set_printoptions(threshold=sys.maxsize)
		print(binary)
		init_boxes = []; # will contain every box which has probability above the detection threashold. 
		for i in range(0,len(binary)):
			this_class = binary[i]
			x,y = positions[i]
			#print("confidence at x %d, y %d: "%(x,y),this_class)
			if this_class<self.DETC_TRHESH:
				cv2.rectangle(trumpim, (x,y),(x+self.IM_WIDTH,y+self.IM_HEIGHT),(0,255,0),1) # draw all windows first. (no NMS)
				init_boxes.append([x,y,x+self.IM_WIDTH,y+self.IM_HEIGHT]);
		print(init_boxes)
		#self.draw_from_threashold(trumpim,init_boxes)

		cv2.imshow("t",trumpim)
		cv2.waitKey()
		cv2.destroyAllWindows()

x = run_one_image()
image = cv2.imread("dude-forest.jpg")
x.detect_faces(image)