import cv2
import sys
import os
import cv2
import numpy as np
from cnn import makeModel, IM_WIDTH, IM_HEIGHT, MODEL_FILENAME
from keras.models import load_model
DETC_TRHESH = 0.05#The threashold for it being a face (higher = more sensitive)

#kernel_size is tupple: (height,width)
#Returns generator. Loop over this function
def make_sliding_window(image, step_size, kernel_size):
	
	image = cv2.resize(image,(new_width, new_height))
	x_up_to = new_width-kernel_size[0]-step_size
	y_up_to = new_height-kernel_size[1]-step_size
	for y in range(0,y_up_to,step_size):
		for x in range(0,x_up_to,step_size):
			yield (x,y,image[y:y+kernel_size[1],x:x+kernel_size[0]])
			#print("x ", x, " y ", y)
image = cv2.imread("trump.jpg")
height, width, depth = image.shape
resize_factor = 0.5
new_height, new_width = (int(height*resize_factor),int(width*resize_factor))
trumpim = cv2.resize(image,(new_width,new_height))
count = 0
images = []
positions = []
for x,y,im in make_sliding_window(trumpim,2, (IM_HEIGHT,IM_WIDTH)):#Don't change kernel size, resize original image instead. Step size can be changed. 
	#cv2.imwrite(os.path.join("sliding_window/",str(count)+".jpg"),im)
	images.append(im)
	positions.append((x,y))
	count = count + 1
	
model =load_model(MODEL_FILENAME)
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
