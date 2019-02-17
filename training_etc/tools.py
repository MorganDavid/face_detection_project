import cv2
import numpy as np
import matplotlib.pyplot as plt

def construct_pyramid(img, num_of_scales):
	scale_by = 1 # start scale
	scale_dec = 0.17 # 
	out_imgs = []
	out_scales = []
	h,w,d = img.shape
	for i in range(num_of_scales):
		new_width = int(w*scale_by)
		new_height = int(h*scale_by)
		scaled_im = cv2.resize(img, (new_width,new_height), cv2.INTER_AREA)
		out_imgs.append(scaled_im)
		out_scales.append(scale_by)
		scale_by -= scale_dec
	return out_imgs, out_scales

#trump = cv2.imread("trump.jpg")
#ims, scales = construct_pyramid(trump,6)

#bounding_boxes in format: 
# (x, y, w ,h)
#returns an image with 0 as not face and 1 as face for every pixel in the image. 
def construct_confidence_matrix(bounding_boxes, img_h, img_w):
	ret_img = np.zeros((img_h, img_w))
	for i in bounding_boxes:
		#convert bounding boxes to 
		top = i[1]
		bottom = i[1]+i[3]
		left = i[0]
		right = i[0]+i[2]
		ret_img[top:bottom, left:right] = 1
	return ret_img
#im = construct_confidence_matrix([(0,0,10,50)],100,100)
#plt.imshow(im, cmap='gray')
#plt.show()

#kernel_size is tupple: (height,width)
#Returns generator. 
def sliding_window_generator(img, stride, kernel_size):
	image = cv2.resize(image,(new_width, new_height))
	x_up_to = new_width-kernel_size[0]-stride
	y_up_to = new_height-kernel_size[1]-stride
	for y in range(0,y_up_to,stride):
		for x in range(0,x_up_to,stride):
			yield (x,y,image[y:y+kernel_size[1],x:x+kernel_size[0]])
			

