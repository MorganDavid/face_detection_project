from widerSetExtraction import * # TODO: change this
import numpy as np
import keras 
from keras.models import load_model
import cv2
import copy 
import sys
import random
_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root

_max_ims = 10000 # number of faces to propose. usually outputs a bit over 
_start_line = 120000
numpy.set_printoptions(threshold=sys.maxsize)

model_dir = '12_net_mtcnn.h5' # the model to mine from
_p_net_model = load_model(model_dir)

dset="train"
_im_save_dir = r"data/48px_mined/{}/neg/".format(dset) # dirs should end in /
_db_save_dir = r"data/48px_mined/"
_pckl_file_name = r"{}_negs.pkl".format(dset)
_net_size = 12
_out_size = 24
# Returns regresion boxes in world coordinates. 
def extract_from_preds(preds, positions, num_of_faces):
	classes = preds[0]
	regr = preds[1]
	# Structure is 3 matrices. one for classificaiton, one for regression and one for position of this box. 
	classes = np.asarray([c[1]-c[0] for c in classes]) # make classes 1 dimensional. Higher more likely it's a face
	k=min(num_of_faces*6,len(classes))
	init_boxes_inds = np.argpartition(classes,-k)[-k:]
	
	init_boxes_pos = np.array(positions)[init_boxes_inds]
	init_boxes_regr =  regr[init_boxes_inds]

	init_boxes = []
	for pos, reg in zip(init_boxes_pos,init_boxes_regr):
		
		x,y,w,h = reg
		w_x, w_y = pos
		init_boxes.append([w_x+x,w_y+y,w,h])
	return init_boxes
'''
kernel_size is tupple: (height,width)
Returns generator. Loop over this function
'''
def make_sliding_window(image, step_size, kernel_size):
	height, width, _ = image.shape
	#image = cv2.resize(image,(width, height))
	x_up_to = width-kernel_size[0]-step_size
	y_up_to = height-kernel_size[1]-step_size
	for y in range(0,y_up_to,step_size):
		for x in range(0,x_up_to,step_size):
			yield (x,y,image[y:y+kernel_size[1],x:x+kernel_size[0]])

def get_names_and_boxes_2(pathToTxt, start_line, max_ims):
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
				if im_count > max_ims: break
				this_bbox =bbxfile[z][:-1]#Take the \n off
				
				string_to_list = [int(i) for i in this_bbox.split()] 
				
				if string_to_list[7] == 0:
					bboxForThisImage.append(string_to_list)
					im_count = im_count+1

	
			bbox.append(bboxForThisImage)
			imgname.append(bbxfile[counter-1][:-1])#-1 to remove the \n
			#print("adding ", len(bboxForThisImage), " faces at line ", counter, " file name is: ", bbxfile[counter-1][:-1])
			if im_count > max_ims: break
			
		counter=counter+1
	print("All ims",imgname)
	return imgname,bbox

'''
Runner method. Scale is the factor we should scale the image by before starting, use scale_image_to_face_size'''
def detect_faces(src_image, scale, num_of_faces):
	height, width, _ = src_image.shape
	image = cv2.resize(src_image, (max(int(width*scale),_net_size*3), max(int(height*scale),_net_size*3)))
	
	count = 0
	images = []
	positions = []
	
	im_iter = make_sliding_window(image,2, (_net_size,_net_size))
	for x,y,im in im_iter:#Don't change kernel size, resize original image instead. Step size can be changed. 
		images.append(im)
		positions.append((x,y))
		count = count + 1
	images = np.asarray(images)
	
	preds = _p_net_model.predict(images, batch_size=50)
	
	init_boxes =  extract_from_preds(preds, positions, num_of_faces)
	return init_boxes
'''
Scales an image to have face height of desired_f_height based on old_f_height
returns new height and width for the image. 
'''
def scale_image_to_face_size(img, desired_f_height, old_f_height):
	height, width, _ = img.shape
	scale_factor = desired_f_height/old_f_height
	n_height, n_width = (int(height*scale_factor), int(width*scale_factor)) 
	return scale_factor, n_width, n_height

#Draws rectangels onto an image with scale factor scale, (use 1 for no scale).
def create_img_with_recs(boxes,image,scale,col):
	new_im = copy.copy(image)
	for box in boxes:
		x,y,x2,y2=[int(n*scale) for n in box]
		#x,y,w,h=box
		cv2.rectangle(new_im,(x,y),(x2,y2),col,1)
	return new_im

#bboxes should be [x1,y1,x2,y2]
def intersection_over_union(bbox1,bbox2):
	b1_x1,b1_y1,b1_x2,b1_y2 = bbox1
	b2_x1,b2_y1,b2_x2,b2_y2 = bbox2

	xinter_l = max(b1_x1,b2_x1) # work out rectangle intersection
	yinter_t = max(b1_y1,b2_y1)
	yinter_b = min(b1_y2,b2_y2)
	xinter_r = min(b1_x2,b2_x2)
	# Check if there is any overlap
	if xinter_r < xinter_l or yinter_b < yinter_t: return 0

	# work out area of rectangle intersection
	inter_area = (yinter_b-yinter_t) * (xinter_r-xinter_l)

	# work out union of both boxes
	bbox1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
	bbox2_area = (b2_x2-b2_x1)*(b2_y2-b2_y1)

	iou = inter_area / float(bbox1_area+bbox2_area-inter_area)
	return iou

#print(intersection_over_union([361, 98, 263, 339],[514, 178, 845, 620]))
img_names_list,bboxes = get_names_and_boxes_2(TRAINING_BBOX_WIDER, _start_line, _max_ims)
print("starting generation, images to generate from: ", len(img_names_list))
out_db = []
im_counter=0
counter = 0
for i,im_name in enumerate(img_names_list):
	this_boxes = [box[:4] for box in bboxes[i]]
	img = cv2.imread(os.path.join(IMAGE_FOLDER_PATH,im_name))
	w,h,_ = img.shape
	# Get average face height and scale image to make it 12 high

	avg_face_height = max(70,np.mean(this_boxes,axis=0)[3] if len(this_boxes) > 1 else this_boxes[0][3]) # the average face height in this image for scaling
	
	scale_factor,new_w,new_h = scale_image_to_face_size(img,random.randint(_net_size,_net_size),avg_face_height) # NET-CHANGE change values of randint to match netsize
	norm_image = cv2.normalize(img, None, alpha=-1, beta=+1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	print("At position:", i, "/",len(img_names_list)," im_name",im_name)

	init_boxes = detect_faces(norm_image, scale=scale_factor,num_of_faces=len(this_boxes))
	
	# convert both boxes arrays to be same format exactly [x1,y1,x2,y2] in image coords
	scale = w/new_w
	init_boxes=[[max(0,int(x*scale)),max(0,int(scale*y)),int(scale*(x+w)),int(scale*(y+h))] for [x,y,w,h] in init_boxes]
	this_boxes=[[x,y,x+w,y+h] for [x,y,w,h] in this_boxes]
	print("init_boxes are ",init_boxes)
	print("this-boxes are ",this_boxes)
	neg_boxes = [] # Just for drawing boxes
	# Get all valid negatives (with appropriately low IOU)
	for y,neg_box in enumerate(init_boxes):
		is_face = False
		print("Currently on init_box: ",neg_box)
		for truth_box in this_boxes:
			print("intersection between {}, and {}".format(truth_box, neg_box))
			iou = intersection_over_union(truth_box,neg_box)
			if iou > 0.02:
				is_face = True 
				print("IS FACE")

		if not is_face: 
			print("not face")
			x1,y1,x2,y2=neg_box
			wid=24
			high=24

			if(x2-x1>wid and y2-y1>high and x2 < w and y2 < h): 
				print(" got here ! ")
				neg_boxes.append(neg_box)
				#n_x = random.randint(x1,x2-wid)
				#n_y = random.randint(y1,y2-high)
				out = img[y1:y2, x1:x2]
				#if out.shape[0] < _out_size or out.shape[1] <_out_size: continue

				out = cv2.resize(out,(wid,high))
				#cv2.imshow("t",out)
				#cv2.waitKey()
				#cv2.destroyAllWindows()
				write_dir = _im_save_dir+"face_"+str(y)+"_"+im_name.split("/")[1]
				success = False
				#success = cv2.imwrite(os.path.join(_root_project_dir,write_dir),out)
				if success:
					counter = counter+1
					out_db.append([write_dir, 0, [0,0,0,0]])
				else:
					print("FAILED WRITE")
	
	out_im = create_img_with_recs(init_boxes,img,1,(0,255,0))
	out_im = create_img_with_recs(neg_boxes,out_im,1,(0,0,255))
	#out_im = create_img_with_recs(this_boxes,out_im,1,(255,0,0))
	cv2.imshow("t",out_im)
	cv2.waitKey()
	cv2.destroyAllWindows()

	im_counter = im_counter+1
'''
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

	'''