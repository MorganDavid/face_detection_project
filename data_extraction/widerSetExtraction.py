import numpy
import cv2
import os
from os.path import dirname, abspath
import random
import itertools # for permutations. 
import math
import pickle 

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root
IMAGE_FOLDER_PATH = os.path.join(_root_project_dir,r"data/raw_datasets/WIDER_train/images") #should be the top level 'images' folder.
TRAINING_BBOX_WIDER = os.path.join(_root_project_dir,r"data/raw_datasets/WIDER_train/wider_face_train_bbx_gt.txt")


# Minimum width and height for every face from the SOURCE dataset. Use _min_target_dim for output size!
# Only one of these needs to be true for image to be valid. 
_min_im_width = 30
_min_im_height = 30

_min_target_dim = 24 # This is the size of the image on x and y, All images are square.

dset = "train" # either train or val

_im_save_dir = r"data/{}px_mined/{}/pos/".format(_min_target_dim,dset) # dirs should end in /
_db_save_dir = r"data/{}px_mined/".format(_min_target_dim)
_pckl_file_name = "{}_pos.pkl".format(dset)

_max_ims = 3500 # the number of images to generate (not faces). 
_start_line = 50000 # the line number to start at if you want to resume from last time. 
_do_extra_transformations = False # Enable this to make every face output multiple images at different transformations and different scales.

_do_write_images = False # Should we write images to file. Turn off for debuging. 
multi_file = True # If this is true then we don't overide old im data, instead add to them.

# Gets all the images from the file. Outputs a list of images, each with a list of boundingboxes inside them. The layout for one bounding box is:
# x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
# See wider readme
def get_names_and_boxes(pathToTxt, start_line, max_ims):
	bbox_file = open(pathToTxt)
	bbxfile = bbox_file.readlines()
	imgname = []
	bbox = []
	im_count = 0 # the number of images we've taken
	counter = 0 # The line in the file we are currently at.
	for i in bbxfile:
		if len(i[:-1])<4 and counter >= start_line: # First check if this is a face count line (length <3)
			bboxForThisImage = []
			for z in range(counter+1,counter+int(i)+1):#Loop over all the boxes in this image. 
				this_bbox =bbxfile[z][:-1]#Take the \n off
				string_to_list = [int(i) for i in this_bbox.split()] 
				#print(string_to_list)
				if string_to_list[2]>=_min_im_width or string_to_list[3]>=_min_im_height: 
					if string_to_list[8] == 0 and string_to_list[7]==0 and string_to_list[9] == 0 and string_to_list[4] == 0 and \
					string_to_list[7] == 0: #8 is occlusion (0 for none, 1 for partial, 2 for full) and 7 is invalid face. 9 is typical vs atypical pose
						bboxForThisImage.append(string_to_list)
						im_count = im_count+1
			
			bbox.append(bboxForThisImage)
			imgname.append(bbxfile[counter-1][:-1])#-1 to remove the \n
			#print("adding ", len(bboxForThisImage), " faces at line ", counter, " file name is: ", bbxfile[counter-1][:-1])
			if im_count > max_ims: break
			
		counter=counter+1
	print(bbox)
	return imgname,bbox

#Extracts a list of faces from all the collected images. 
def extract_list_of_faces(img_names_list,bboxes):
	print ("Starting face extraction from file!" if _do_write_images else "[NOT WRITING IMS TO FILE!] Starting face extraction")
	im_counter = 0 # counts what image we are currenlty looking at.
	face_imgs = []
	face_counter = 0 # Counts how many faces we got so far 
	out_db = [] # This is the array that we will pickle out. Structure [[img_path,label,[x1,y1,w,h]]]

	for im_name in img_names_list:
		img = cv2.imread(os.path.join(IMAGE_FOLDER_PATH,im_name))	
		width, height, _ = img.shape
		print("at ",im_name)
		for box in bboxes[im_counter]: # Loop through all the bboxes in this image. 
			trans_x = [-2,-1,0,1,2]
			trans_y = [-4,-3,-2,-1,0,1,2,3,4]
			transfrms = [list(x) for x in itertools.product(trans_x,trans_y)]
			transfrms = numpy.asarray(transfrms)[numpy.random.randint(0,len(trans_x)*len(trans_y),size=random.randint(1,7))]
			#print("transforms are:",transfrms)
			if not _do_extra_transformations: transfrms = [[0,0]]
			for trans_x,trans_y in transfrms:
				box = numpy.asarray(box)
				box = box[:4].tolist()
				x,y,w,h = box
				real_im_name = im_name.split("/")[1]
				#print("this sbox is: ", box)
				
				transform_range = 0.2
				# Make new size randomly selected based on face size. 
				new_size = random.randint(int( min(w,h)*0.9),int(max(w,h)*1.3))
				x_trans = random.randint(int(w*-transform_range), int(w*transform_range))+trans_x # transform the box based on width, randomly. 
				y_trans = random.randint(int(h*-transform_range), int(h*transform_range))+trans_y

				# x,y centre of the bbox
				x_centre = box[0] + w/2
				y_centre = box[1] + h/2

				# Tranformed centre
				new_x_centre = x_centre + x_trans
				new_y_centre = y_centre + y_trans
				# new top left corner
				new_x = int(new_x_centre - new_size/2)
				new_y = int(new_y_centre - new_size/2)
				new_w = new_size 
				new_h = new_size 

				if new_x < 0 or new_y < 0 or new_w+new_x > width or new_h+new_y > height: continue

				crop_area_im = img[new_y:new_size+new_y,new_x:new_x+new_size]
				crp_w,crp_h, _ = crop_area_im.shape
			
				# Construct the coordinates of this face bbox relative to new crop_area_im
				b_x = x-new_x
				b_y = y-new_y
				new_bbox = [b_x,b_y,w,h] 

				# Work out the resize factor so the sizes are relative to the new small image. 
				x_scale = _min_target_dim/crp_w
				y_scale = _min_target_dim/crp_h
				#Convert the new bbox to new scale
				scaled_bbox = [int(new_bbox[0]*x_scale), int(new_bbox[1]*y_scale), int(new_bbox[2]*x_scale), int(new_bbox[3]*y_scale)] # stored as [x,y,w,h]

				resized_im = cv2.resize(crop_area_im,(_min_target_dim,_min_target_dim),interpolation=cv2.INTER_CUBIC)
				#cv2.rectangle(resized_im, (scaled_bbox[0],scaled_bbox[1]),(scaled_bbox[0]+scaled_bbox[2],scaled_bbox[1]+scaled_bbox[3]),(0,255,0))# enable this to draw the bounding boxes on the output. 

				im_path = _im_save_dir+str(_start_line + face_counter)+".jpg"
				done = cv2.imwrite(os.path.join(_root_project_dir, im_path), resized_im)

				out_db.append([im_path, 1, scaled_bbox]) # Add this face output database.
				face_counter = face_counter+1
		im_counter = im_counter+1
	print("Generated im count: ", len(out_db) )
	# If there is already a file with equal name, append to that. 
	if multi_file and os.path.isfile(os.path.join(_root_project_dir,_db_save_dir+_pckl_file_name)):
		print("MULTI FILE")
		with open(os.path.join(_root_project_dir,_db_save_dir+_pckl_file_name),'rb') as f:
			old = pickle.load(f)
		out_db = out_db+old
	print("new length of db is",len(out_db))

	with open(os.path.join(_root_project_dir,_db_save_dir+_pckl_file_name),'wb') as f2:
		pickle.dump(out_db, f2)
	with open(os.path.join(_root_project_dir,_db_save_dir+_pckl_file_name),'rb') as f3:
		check = pickle.load(f3)
	print("writen database has ", len(check))
	

	return face_imgs

# Returns a list of FULL (not just the bbox) images for this bbox. scales is the list of face heights you want.
# output is [img, x,y,w,h] where the coords are the new scaled coords of this bounding box.  
def make_img_pyramid(im, bbox,scales = [18,19,20,22]):
	scaled_imgs = []
	width, height, _ = im.shape
	x,y,w,h = bbox
	for scale in scales:
		scale_amnt = scale / min(w,h) # How much do we need to resize by to get our min target dimension.
		img_resized = cv2.resize(im,(int(height*scale_amnt),int(width*scale_amnt)))
		new_x,new_y,new_w,new_h = map(lambda x: math.floor(x*scale_amnt),[x,y,w,h])# also return the adjusted bbox sizes. 
		scaled_imgs.append((img_resized,new_x,new_y,new_w,new_h))
	return scaled_imgs

#Opens an image viewer with bounding boxes drawn over it. 
#bbox should be a list, while image_path should be a string path
def show_image_with_bboxes(image_path,bbox):
	#bbox = bbox[]#only need the first four
	print(bbox)
	img = cv2.imread(os.path.join(IMAGE_FOLDER_PATH,image_path))

	#Drawing bounding boxes onto image
	for box in bbox:
		cv2.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),1)
	#show image with bounding boxes
	cv2.imshow("image", img)
	cv2.waitKey()
if '__main__' == __name__:
	names,bboxes = get_names_and_boxes(TRAINING_BBOX_WIDER, _start_line, _max_ims)
	print("FINSISHED: Gotten ", len(bboxes), " bboxes from ",len(names), " images.") 
	face_list = extract_list_of_faces(names,bboxes)
	print("FINISHED: finished extracting faces from ",len(names)," images")
	#show_image_with_bboxes(names[24],bboxes[24])

