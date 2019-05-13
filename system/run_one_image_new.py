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
	_p_det_thresh = -0.959608#The threashold for it being a face (higher = more sensitive) This is used when we need to detect more than one face in the image.harry: 1.5e11 1d: 1.5e4
	_r_det_tresh = 0.9999
	NUM_FACES = 10 # This is used instead of DETEC_THREASH. Use the top_k_faces function instead. NUM_FACES=2 means take the top 2 most confident face boxes and draw them.
	IM_WIDTH = 12 # this actually the dims the CNN was trained on. 
	IM_HEIGHT = 12
	p_net_model_dir = r'P-net-mining.h5'
	r_net_model_dir = r'R-net-mining.h5'
	SEGMENT_METHOD = "f" # either f or q
	SEGMENT_MAX_RECTS = 200 # number of segments to produce
	USE_SEGMENTATION = False # True to use selective search segmentation instead of sliding window. 
	_p_net_model = 1
	_r_net_model = 1
	def __init__(self):
		self._p_net_model = load_model(self.p_net_model_dir)
		self._r_net_model = load_model(self.r_net_model_dir)
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
	# Disclaimer: This function was taken from https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
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
		boxes = boxes[pick].astype("int")
		#convert to x,y,w,h instead of x1,y1, etc.
		boxes = [[x,y,x2-x,y2-y] for (x,y,x2,y2) in boxes]
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
	
	#This method takes the top K confidence boxes and uses them.
	def draw_top_k_rectangles(self, img, classes, positions,k):
		indices = np.argsort(classes.flatten())[-k:]#np.argpartition(classes.flatten(),k)[:k]
		boxes_top_k_xy = [positions[i] for i in indices] # get x,y of every box
		boxes_top_k = [[i[0],i[1],self.IM_WIDTH,self.IM_HEIGHT] for i in boxes_top_k_xy]
		return boxes_top_k

	def extract_from_r_preds(self,preds, positions): # For models with normalized regression output.
		classes = preds[0]
		regr = preds[1]
		classes = np.asarray([c[1]-c[0] for c in classes]) # make classes 1 dimensional. Higher more likely it's a face
		init_boxes_inds = np.argwhere(classes>self._r_det_tresh)
		print("inds are",init_boxes_inds)
		print("regr {}. \n\n clas {}".format(regr,classes))
		init_boxes_pos = [x[0] for x in np.array(positions)[init_boxes_inds]] # for loop flattens away 1 dimension
		init_boxes_regr = [x[0] for x in regr[init_boxes_inds]]
		init_boxes = []
		print("init_boxes_pos are:", init_boxes_pos)
		print("regr",init_boxes_regr)

		for pos, reg in zip(init_boxes_pos,init_boxes_regr):
			#print("old reg",reg)
			#reg *= 24 # only enable this if model is normalised
			#print("new reg",reg)
			x,y,w,h = reg
			w_x, w_y = pos
			init_boxes.append([w_x+x,w_y+y,w,h])
		#print("R-net init_boxes",init_boxes)
		return init_boxes

	# Returns regresion boxes in world coordinates. 
	def extract_from_p_preds(self, preds, positions):
		classes = preds[0]
		regr = preds[1]
		# Structure is 3 matrices. one for classificaiton, one for regression and one for position of this box. 
		classes = np.asarray([c[1]-c[0] for c in classes]) # make classes 1 dimensional. Higher more likely it's a face
		init_boxes_inds = np.argwhere(classes>self._p_det_thresh)
		#print("classes pnet:",is_pnet,":",classes)

		init_boxes_pos = [x[0] for x in np.array(positions)[init_boxes_inds]] # for loop flattens away 1 dimension
		init_boxes_regr = [x[0] for x in regr[init_boxes_inds]]
		init_boxes = []

		#if not is_pnet: 
		#	init_boxes_regr = [[min(1,max(0,x)),min(1,max(0,y)),min(1,max(0,w)),min(1,max(0,h))] for x,y,w,h in init_boxes_regr]
			
		#print("regr boxes are:", init_boxes_regr)
		for pos, reg in zip(init_boxes_pos,init_boxes_regr):
			x,y,w,h = reg
			w_x, w_y = pos
			box = [w_x+x,w_y+y,w,h]
			init_boxes.append(box)
		return init_boxes
	def norm_and_show_im(self,title,im): # debug function. show image half done
		disp_im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
		
		cv2.imshow(title,disp_im.astype(np.uint8))
		cv2.waitKey()
		cv2.destroyAllWindows()
	'''
	Runner method. Scale is the factor we should scale the image by before starting, use scale_image_to_face_size'''
	def detect_faces(self, src_image, scale):
		height, width, _ = src_image.shape
		image = cv2.resize(src_image, (int(width*scale), int(height*scale)))
		big_image = cv2.resize(src_image, (int(width*scale*2),int(height*scale*2))) # for 24px R-net
		#print("resized image to: ", image.shape)
		count = 0
		images = []
		positions = []
		#print("starting the sliding window loop. ")
		start = time.time()
		blahblah = 0
		if self.USE_SEGMENTATION:
			im_iter = self.get_image_segmentations(image)
		else:
			im_iter = self.make_sliding_window(image,2, (self.IM_HEIGHT,self.IM_WIDTH))
		for x,y,im in im_iter:#Don't change kernel size, resize original image instead. Step size can be changed. 
			images.append(im)
			positions.append([x,y])
			count = count + 1
			blahblah=blahblah+1
		print("NUMBER OF SLIDES ",blahblah)
		#print("sliding window took ",time.time()-start, " seconds.")	
		images = np.asarray(images)

		#print("starting P-net prediction")
		start = time.time()
		preds = self._p_net_model.predict(images, batch_size=50)
		#print("P-net prediction took ",time.time()-start, " seconds.")

		init_boxes =  self.extract_from_p_preds(preds, positions)
		old_init_boxes = init_boxes
		print("starting R-net prediction")
		start = time.time()
		print("init boxes before thing0",init_boxes)
		ims_for_rnet = []
		for box in init_boxes:
			x,y,w,h = box
			x,y,w,h = [int(x) for x in [x*2,y*2,w*2,h*2]]
			if w < 5 or h < 5: continue
			im = big_image[max(0,y):min(y+h,big_image.shape[0]),max(0,x):min(x+w,big_image.shape[1])]
			#self.norm_and_show_im("blah",im)
			im = cv2.resize(im,(24,24))
			
			ims_for_rnet.append(im)
		ims_for_rnet = np.asarray(ims_for_rnet)
		print(init_boxes)
		positions = np.asarray(init_boxes)[:,:2]
		print("postions",positions)
		r_preds = self._r_net_model.predict(ims_for_rnet, batch_size = 50)

		init_boxes = self.extract_from_r_preds(r_preds,positions)
		#for i,box in enumerate(init_boxes):
		#	for x,num in enumerate(box):
		#		init_boxes[i][x] = int(num)
		#print("init_boxes from R-net are", init_boxes)
		#print("R-net took ",time.time()-start, " seconds")

		boxes_nms = self.non_max_suppression(np.asarray([[x,y,x+w,y+h] for [x,y,w,h] in init_boxes]),0.28)

		return old_init_boxes, init_boxes, boxes_nms # pnet, rnet, nms
		
	#Drawss rectangels onto an image with scale factor scale, (use 1 for no scale).
	def create_img_with_recs(self,boxes,image,scale):
		new_im = copy.copy(image)
		for box in boxes:
			x,y,w,h=[int(n*scale) for n in box]
			#x,y,w,h=box
			cv2.rectangle(new_im,(x,y),(x+w,y+h),(0,255,0),2)
		return new_im
		
if __name__ == "__main__":
	_final_stage_resize_factor = 1 # This for degbugging. =0.5 means the cv2.imshow will half the image size and boxes size.
	x = image_predictor()

	image = cv2.imread("dude-forest.jpg")
	norm_image = cv2.normalize(image, None, alpha=-1, beta=+1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	old_height, old_width, _ = image.shape

	scale_factor, n_width, n_height = x.scale_image_to_face_size(image, 50, 200)

	pnet_boxes, rnet_boxes, boxes = x.detect_faces(norm_image,scale_factor)

	#image = cv2.resize(image,(int(old_width*_final_stage_resize_factor),int(old_height*_final_stage_resize_factor)))
	final_im = x.create_img_with_recs(rnet_boxes,image,int((old_width)/n_width))#int((old_width*0.5)/n_width))

	cv2.imshow("t",final_im)
	cv2.waitKey()
	cv2.destroyAllWindows()
