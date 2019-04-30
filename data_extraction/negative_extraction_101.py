import os
from os.path import dirname, abspath
import cv2
from widerSetExtraction import get_names_and_boxes
import random
import numpy as np
import math 

_min_target_dim = 24

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root
print(_root_project_dir)
_images_path = os.path.join(_root_project_dir, r"data/raw_datasets/101_ObjectCategories")

categories = os.listdir(_images_path)

for i in range(6,len(categories)):

	cat = categories[i]
	print(cat)

	for im_name in os.listdir(os.path.join(_images_path,cat)):

		im_pth = os.path.join(_images_path,os.path.join(cat,im_name))
		print(im_pth)
		im = cv2.imread(im_pth)
		height, width, _ = im.shape
		print(im_pth)
		#randomly calculate the possition of this negative box
		neg_width = _min_target_dim
		neg_height =_min_target_dim
		neg_x = random.randint(0,width-neg_width)
		neg_y = random.randint(0,height-neg_height)
		cropped_img = im[neg_y:neg_y+neg_height,neg_x:neg_x+neg_width]
		cv2.imwrite(os.path.join("101negs/",str(random.randint(1,10000))+im_name),cropped_img)
