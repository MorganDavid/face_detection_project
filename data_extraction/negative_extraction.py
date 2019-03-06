import os
from os.path import dirname, abspath
import cv2
from widerSetExtraction import get_names_and_boxes
import random
import numpy as np
import math 

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root
print(_root_project_dir)
_images_path = os.path.join(_root_project_dir, r"data/raw_datasets/WIDER_train/images")
_txt_path = os.path.join(_root_project_dir,r"data/raw_datasets/WIDER_train/wider_face_train_bbx_gt.txt")

_max_ims = 500
_min_target_dim = 20 # make sure this is the same as the var in the widerSetExtraction file. Determines the minimum image dimension of the output.
_start_line =115000
_negatives_per_image_max = 3

# takes the list of bboxes and removes all but the positional attributes (the first four columns). 
def reduce_cols_from_bboxes(boxes):
	for i in range(0,len(boxes)):
			b = boxes[i]
			if len(b)>0:
				b = np.asarray(b)
				boxes[i] = b[:,:4].tolist()
	return boxes

# Gets the negative images (not faces) from the dataset. Use get_names_and_boxes 
# from widerSetExtraction for names and boxes variables. 
def extract_negatives(image_path, names, boxes, negs_per_img):
	file_cnt = 0 # counts which image we are on right now. 
	negs_to_return = [] # the array of images we return. 

	boxes = reduce_cols_from_bboxes(boxes)
	
	for file in names:
		img = cv2.imread(os.path.join(_images_path,file))	
		height,width, _ = img.shape

		negs_per_this_img = random.randint(1,negs_per_img) # randomly calculate how many negatives to make from this image. 
		#print("boxes is ", boxes)
		for i in range(0,negs_per_this_img):
			iou = 100
			# keep generating boxes until we get a box that is less than the IOU threshold. 
			# TODO: make IOU value a percentage of image size (not sure if needed) currently using number of pixels overlapped
			while(iou>2):
				#randomly calculate the possition of this negative box
				neg_width = _min_target_dim
				neg_height = random.randint(_min_target_dim,_min_target_dim+10)
				neg_x = random.randint(0,width-neg_width)
				neg_y = random.randint(0,height-neg_height)

				iou = iou_between_list([neg_x, neg_y, neg_width, neg_height], boxes[file_cnt])
				#print("iou: ", iou)
			cropped_img = img[neg_y:neg_y+neg_height,neg_x:neg_x+neg_width]
			print("got %s from %s"%(cropped_img.shape,file))
			#print("file: %s w %d h %d x %d y %d im_width %d im_height %d file_count: %d generated image shape %s"%(file,neg_width, neg_height,neg_x,neg_y,width,height,file_cnt, cropped_img.shape))
			negs_to_return.append(cropped_img)
			
		file_cnt=file_cnt+1
	return negs_to_return

#check the maximum iou value between bbox1 and all img_bboxes.
def iou_between_list(bbox1, img_bboxes):
	iou_max = 0
	#print("bbox1 is: ", bbox1)
	for b in img_bboxes: # loops through every image.
		iou = intersection_over_union(bbox1, b)
		
		if iou > iou_max: 
			print("%s intersects with %s"%(b,bbox1))
			iou_max = iou
	return iou_max


# Gets the IOU between two rectangles.
# Taken from https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
def intersection_over_union(bbox1,bbox2):
	x1, y1, w1, h1 = bbox1
	x2, y2, w2, h2 = bbox2

	x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2));
	y_overlap = max(0, min(y1+h1, y2+h2) - max(y1,y2));
	overlapArea = x_overlap * y_overlap;
	return overlapArea


names, boxes = get_names_and_boxes(_txt_path, _start_line, _max_ims) # use functoin from wider file. Just gets all filenames linked to bounding boxes. 
#print(names, " ", boxes)

negatives = extract_negatives(_images_path, names, boxes, _negatives_per_image_max)
for n in range(0,len(negatives)):
	write_dir = os.path.join(_root_project_dir,r"data/negatives20px/rand")+str(random.randint(0,10))+str(names[n].split('/')[1])
	cv2.imwrite(write_dir,negatives[n])
	print("Just wrote somin to ",write_dir)