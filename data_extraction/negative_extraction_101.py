import os
from os.path import dirname, abspath
import cv2
from widerSetExtraction import get_names_and_boxes
import random
import numpy as np
import math 
import pickle 

_min_target_dim = 24
_max_ims = 100
_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root
print(_root_project_dir)
_images_path = os.path.join(_root_project_dir, r"data/raw_datasets/101_ObjectCategories")
dset = "test"
_im_save_dir = r"data/{}px_new5k/{}/neg/".format(_min_target_dim,dset) # dirs should end in /
_db_save_dir = r"data/{}px_new5k/".format(_min_target_dim)
_pckl_file_name = r"test_negs.pkl"
categories = os.listdir(_images_path)
negatives = []
ims_cnt = 0
for i in range(6,len(categories)):

	cat = categories[i]
	
	if ims_cnt >= _max_ims: break
	for im_name in os.listdir(os.path.join(_images_path,cat)):

		im_pth = os.path.join(_images_path,os.path.join(cat,im_name))
		im = cv2.imread(im_pth)
		height, width, _ = im.shape
		#randomly calculate the possition of this negative box
		neg_width = _min_target_dim
		neg_height =_min_target_dim
		neg_x = random.randint(0,width-neg_width)
		neg_y = random.randint(0,height-neg_height)
		cropped_img = im[neg_y:neg_y+neg_height,neg_x:neg_x+neg_width]
		negatives.append(cropped_img)
		ims_cnt= ims_cnt + 1 
		print(ims_cnt)
		if ims_cnt >= _max_ims: break
print("Starting writer")
num=0
out_db = [] # file to be pickled contain all file paths and boxes. 
for n in range(0,len(negatives)):
	write_dir = _im_save_dir+str((num))+".jpg"
	out_db.append([write_dir, 0, [0,0,0,0]])

	cv2.imwrite(os.path.join(_root_project_dir,write_dir),negatives[n])
	print("Just wrote somin to ",write_dir)
	num = num + 1

# If there is already a file with equal name, append to that. 
if os.path.isfile(os.path.join(_root_project_dir,_db_save_dir+_pckl_file_name)):
	print("MULTI FILE")
	with open(os.path.join(_root_project_dir,_db_save_dir+_pckl_file_name),'rb') as f:
		old = pickle.load(f)
	out_db = out_db+old
print("new length of db is",len(out_db))

with open(os.path.join(_root_project_dir,_db_save_dir+_pckl_file_name),'wb') as f2:
	pickle.dump(out_db, f2)
with open(os.path.join(_root_project_dir,_db_save_dir+_pckl_file_name),'rb') as f3:
	check = pickle.load(f3)
	print("writen as ", check)
print("writen database has ", len(check))
	