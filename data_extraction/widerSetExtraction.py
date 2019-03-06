import numpy
import cv2
import os
from os.path import dirname, abspath
import random
import itertools # for permutations. 
import math

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root
IMAGE_FOLDER_PATH = os.path.join(_root_project_dir,r"data/raw_datasets/WIDER_train/images") #should be the top level 'images' folder.
TRAINING_BBOX_WIDER = os.path.join(_root_project_dir,r"data/raw_datasets/WIDER_train/wider_face_train_bbx_gt.txt")

# Minimum width and height for every face from the SOURCE dataset. Use _min_target_dim for output size!
# Only one of these needs to be true for image to be valid. 
_min_im_width = 20 
_min_im_height = 20

_min_target_dim = 20 # the smallest dimension of the output images. 

_max_ims = 20 # the number of images to generate. 
_start_line = 100000 # the line number to start at if you want to resume from last time. 

_do_write_images = True # Should we write images to file. Turn off for debuging. 

# Gets all the images from the file. Outputs a list of images, each with a list of boundingboxes inside them. The layout for one bounding box is:
# x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
# See wider readme
def get_names_and_boxes(pathToTxt, start_line, max_ims):
	bbox_file = open(pathToTxt)
	bbxfile = bbox_file.readlines()
	imgname = []
	bbox = []
	im_count = 0 # the number of images we've taken
	counter = 0#The line in the file we are currently at.
	for i in bbxfile:
		if len(i[:-1])<4 and counter >= start_line:#First check if this is a face count line (length <3)
			bboxForThisImage = []
			for z in range(counter+1,counter+int(i)+1):#Loop over all the boxes in this image. 
				this_bbox =bbxfile[z][:-1]#Take the \n off
				string_to_list = [int(i) for i in this_bbox.split()] 
				#print(string_to_list)
				if string_to_list[2]>=20 or string_to_list[3]>=20: 
					if string_to_list[8] == 0 and string_to_list[7]==0 and string_to_list[9] == 0 and string_to_list[4] == 0 and \
					string_to_list[7] == 0: #8 is occlusion (0 for none, 1 for partial, 2 for full) and 7 is invalid face. 9 is typical vs atypical pose
						bboxForThisImage.append(string_to_list)
						im_count = im_count+1
			
			bbox.append(bboxForThisImage)
			imgname.append(bbxfile[counter-1][:-1])#-1 to remove the \n
			#print("adding ", len(bboxForThisImage), " faces at line ", counter, " file name is: ", bbxfile[counter-1][:-1])
			if im_count > max_ims: break
			
		counter=counter+1
	return imgname,bbox

#Extracts a list of faces from all the collected images. 
def extract_list_of_faces(img_names_list,bboxes):
	print ("Starting face extraction from file!" if _do_write_images else "[NOT WRITING IMS TO FILE!] Starting face extraction")
	im_counter = 0 # counts what image we are currenlty looking at.
	face_imgs = []
	for im_name in img_names_list:
		img = cv2.imread(os.path.join(IMAGE_FOLDER_PATH,im_name))	
		width, height, _ = img.shape
		face_counter = 0 # Counts how many faces we got from this image. 
		
		for box in bboxes[im_counter]: # Loop through all the bboxes in this image. 
			box = numpy.asarray(box)
			box = box[:4].tolist()

			real_im_name = im_name.split("/")[1]
			print("this sbox is: ", box)
			
			x,y,w,h = box
			
			scale_amnt = 12 / min(w,h) # How much do we need to resize by to get our min target dimension.
			img_resized = cv2.resize(img,(int(height*scale_amnt),int(width*scale_amnt)))
			new_width, new_height, _ = img_resized.shape
			
			x,y,w,h = map(lambda x: math.floor(x*scale_amnt),[x,y,w,h]) # convert all the coords to the resized image. 

			#print("x is %d, y is %d. img dims are: %d x %d "%(x,y,new_width, new_height))
			
			#face = img_resized[y:y+h,x:x+w]
			# -- transform the bounding box around the face. --
			transformations_x = [-1,0,1] # amounts to move the box around the face for preventing overfitting.
			transformations_y = [-3,-2,-1,0,1,2] 
			perms = list(itertools.product(transformations_x,transformations_y))
			#print("permutations are: ", perms, ". creating ", len(perms), " images for this face.")
			
			for trans_x,trans_y in perms:
				face = img_resized[y+trans_y:y+trans_y+h,x+trans_x:x+trans_x+w]
				#cv2.imshow("a",face)
				#cv2.waitKey()

				if _do_write_images: cv2.imwrite(os.path.join(os.path.join(_root_project_dir,r"data/20px/positives/")+str(face_counter)+"_"+real_im_name),face) 
				
				face_counter = face_counter+1

			#print(face_counter," faces done already. just wrote face ", im_name+"- x: ", x, " y: ", y, " w: ", w, " h: ", h)
		print("finished writing image: ",im_name)
		im_counter = im_counter+1
	return face_imgs

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

