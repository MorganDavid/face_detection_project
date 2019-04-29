import cv2
import sys
import os
import cv2
import numpy as np
from keras.models import load_model
import time
from image_segmenter import image_segmenter
import copy

class image_predictor():
	DETC_TRHESH = 1.5e-8#The threashold for it being a face (higher = more sensitive) This is used when we need to detect more than one face in the image.harry: 1.5e11 1d: 1.5e4
	NUM_FACES = 6 # This is used instead of DETEC_THREASH. Use the top_k_faces function instead. NUM_FACES=2 means take the top 2 most confident face boxes and draw them.
	IM_WIDTH = 12 # this actually the dims the CNN was trained on. 
	IM_HEIGHT = 12
	_model_dir = r'test.h5'
	SEGMENT_METHOD = "f" # either f or q
	SEGMENT_MAX_RECTS = 200 # number of segments to produce
	USE_SEGMENTATION = False # True to use selective search segmentation instead of sliding window. 
	_model = 1
	def __init__(self):
		self._model = load_model(self._model_dir)
	'''
	kernel_size is tupple: (height,width)
	Returns generator. Loop over this function
	'''
	def make_sliding_window(self, image, step_size, kernel_size):
		height, width, _ = image.shape
		#image = cv2.resize(image,(width, height))
		x_up_to = width-kernel_size[0]-step_size
		y_up_to = height-kernel_size[1]-step_size
		for y in range(0,y_up_to,step_size):
			for x in range(0,x_up_to,step_size):
				yield (x,y,image[y:y+kernel_size[1],x:x+kernel_size[0]])
	
	def get_image_segmentations(self, image):
		sgmntr = image_segmenter()
		retList = sgmntr.segment_image(image, self.SEGMENT_METHOD, self.SEGMENT_MAX_RECTS)
		for x, y, im in retList:
			w, h, _ = im.shape
			wh_ratio = w/h # Make sure the segmentation is roughly square. 
			if w > 10 and h > 10 and wh_ratio < 3 and wh_ratio > 0.33:
				yield (x,y,im)

	'''
	Scales an image to have face height of desired_f_height based on old_f_height
	returns new height and width for the image. 
	'''
	def scale_image_to_face_size(self, img, desired_f_height, old_f_height):
		height, width, _ = img.shape
		scale_factor = desired_f_height/old_f_height
		n_height, n_width = (int(height*scale_factor), int(width*scale_factor)) 
		return scale_factor, n_width, n_height

	# Malisiewicz et al.
	# This function was taken from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	def non_max_suppression(self, boxes, overlapThresh):
		print("boxes are ", boxes)
		
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
		boxes = boxes[pick].astype("int")
		#convert to x,y,w,h instead of x1,y1, etc.
		boxes = [[x,y,x2-x,y2-y] for (x,y,x2,y2) in boxes]
		print("boxes are ", boxes)
		return boxes
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
	
	#This method takes the top K confidence boxes and uses them. Returns as 4 COORDS, NOT W, H.
	def draw_top_k_rectangles(self, img, classes, positions,k):
		indices = np.argpartition(classes.flatten(),k)[:k]
		boxes_top_k_xy = [positions[i] for i in indices] # get x,y of every box
		boxes_top_k = [[i[0],i[1],i[0]+self.IM_WIDTH,i[1]+self.IM_HEIGHT] for i in boxes_top_k_xy] # make them four coords. 
		return boxes_nms

	'''
	Runner method. Scale is the factor we should scale the image by before starting, use scale_image_to_face_size'''
	def detect_faces(self, image, scale):
		height, width, _ = image.shape
		image = cv2.resize(image, (int(width*scale), int(height*scale)))
		print("resized image to: ", image.shape)
		count = 0
		images = []
		positions = []
		print("starting the sliding window loop. ")
		start = time.time()
		
		if self.USE_SEGMENTATION:
			im_iter = self.get_image_segmentations(image)
		else:
			im_iter = self.make_sliding_window(image,2, (self.IM_HEIGHT,self.IM_WIDTH))
		for x,y,im in im_iter:#Don't change kernel size, resize original image instead. Step size can be changed. 
			images.append(im)
			positions.append((x,y))
			count = count + 1
			

		print("sliding window took ",time.time()-start, " seconds.")	
		images = np.asarray(images)

		print("starting prediction")
		start = time.time()
		model = self._model
		classes = model.predict(images, batch_size=50)
		print("prediction took ",time.time()-start, " seconds.")
		
		init_boxes = []; # will contain every box which has probability above the detection threashold. 
		for i in range(0,len(classes)):
			this_class = classes[i]
			x,y = positions[i]
			print("confidence at x %d, y %d: "%(x,y),this_class)
			if this_class<self.DETC_TRHESH:
				print("box at ",x,", ",y)
				cv2.rectangle(image, (x,y),(x+self.IM_WIDTH,y+self.IM_HEIGHT),(255,0,0),1) # draw all windows first. (no NMS)
				init_boxes.append([x,y,self.IM_WIDTH,self.IM_HEIGHT])

		top_k_boxes = self.draw_top_k_rectangles(image,classes,positions,self.NUM_FACES)
		boxes_nms = self.non_max_suppression(np.asarray(top_k_boxes),0.28)
		return init_boxes, boxes_nms
		
	#Drawss rectangels onto an image with scale factor scale, (use 1 for no scale).
	def create_img_with_recs(self,boxes,image,scale):
		new_im = copy.copy(image)
		for box in boxes:
			x,y,w,h=[int(n*scale) for n in box]
			#x,y,w,h=box
			cv2.rectangle(new_im,(x,y),(x+w,y+h),(0,255,0),1)
		return new_im
		
if __name__ == "__main__":
	_final_stage_resize_factor = 1 # This for degbugging. =0.5 means the cv2.imshow will half the image size and boxes size.
	x = image_predictor()

	image = cv2.imread("women.jpg")
	norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	old_height, old_width, _ = image.shape

	scale_factor, n_width, n_height = x.scale_image_to_face_size(image, 50, 2000)

	init_boxes, boxes = x.detect_faces(norm_image,scale_factor)
	print("initboxes are:", init_boxes)
	#image = cv2.resize(image,(int(old_width*_final_stage_resize_factor),int(old_height*_final_stage_resize_factor)))
	final_im = x.create_img_with_recs(init_boxes,image,int((old_width)/n_width))#int((old_width*0.5)/n_width))
	cv2.imshow("t",final_im)
	cv2.waitKey()
	cv2.destroyAllWindows()
