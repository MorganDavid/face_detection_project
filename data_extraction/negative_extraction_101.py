import os
from os.path import dirname, abspath
import cv2
from widerSetExtraction import get_names_and_boxes
import random
import numpy as np
import math 

_min_target_dim = 20

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root
print(_root_project_dir)
_images_path = os.path.join(_root_project_dir, r"data/20px/test/negatives")

im_names = os.listdir(_images_path)

for im_name in im_names:
	im_pth = os.path.join(_images_path, im_name)
	im = cv2.imread(im_pth)
	height, width, _ = im.shape
	print(im_pth)
	#randomly calculate the possition of this negative box
	neg_width = _min_target_dim
	neg_height = random.randint(_min_target_dim,_min_target_dim+10)
	neg_x = random.randint(0,width-neg_width)
	neg_y = random.randint(0,height-neg_height)
	cropped_img = im[neg_y:neg_y+neg_height,neg_x:neg_x+neg_width]
	cv2.imwrite(os.path.join("101negs/",im_name),cropped_img)