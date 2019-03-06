import cv2
import sys
import os
import cv2
import numpy as np
from cnn import makeModel,  MODEL_FILENAME
from keras.models import load_model
DETC_TRHESH = 1.5e-10#0.05#The threashold for it being a face (higher = more sensitive)

IM_WIDTH = 20
IM_HEIGHT = 20

#kernel_size is tupple: (height,width)
#Returns generator. Loop over this function
def make_sliding_window(image, step_size, kernel_size):
	
	image = cv2.resize(image,(new_width, new_height))
	x_up_to = new_width-kernel_size[0]-step_size
	y_up_to = new_height-kernel_size[1]-step_size
	for y in range(0,y_up_to,step_size):
		for x in range(0,x_up_to,step_size):
			yield (x,y,image[y:y+kernel_size[1],x:x+kernel_size[0]])
			
image = cv2.imread("scaled_image_harry.jpg")
height, width, depth = image.shape
print(image.shape)
resize_factor = 0.5
new_height, new_width = (int(height*resize_factor),int(width*resize_factor))
trumpim = cv2.resize(image,(new_width,new_height))
count = 0
images = []
positions = []
print("starting the sliding window loop. ")
for x,y,im in make_sliding_window(trumpim,2, (IM_HEIGHT,IM_WIDTH)):#Don't change kernel size, resize original image instead. Step size can be changed. 
	#cv2.imshow("t",im)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	print("looping through sliding window")
	images.append(im)
	positions.append((x,y))
	count = count + 1
	
model =load_model("test_model.h5")
images = np.asarray(images)
print(images.shape)

classes = model.predict(images, batch_size=10)
for i in range(0,len(classes)):
	this_class = classes[i]
	x,y = positions[i]
	print(i, '.jpg : ', this_class)
	if this_class<DETC_TRHESH:
		cv2.rectangle(trumpim, (x,y),(x+IM_WIDTH,y+IM_HEIGHT),(0,255,0),2)
cv2.imshow("t",trumpim)
cv2.waitKey()
cv2.destroyAllWindows()
