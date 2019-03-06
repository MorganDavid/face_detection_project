from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Conv2D, BatchNormalization
from tools import construct_pyramid, sliding_window_generator

# first network in the cascade
def make_12net(window_size = 12):
	model = Sequential()
	model.add(Conv2D(16,(3,3), strides=1, input_shape=(window_size,window_size,3)))
	model.add(MaxPooling2D((3,3),strides=2))
	model.add(Flatten())
	model.add(Dense(16))
	model.add(Dense(1, activation='softmax'))
	return model

# second network in the cascade
def make_24net(window_size = 24):
	model = Sequential()
	model.add(Conv2D(64,(5,5),strides=1, input_shape=(window_size,window_size,3)))
	model.add(MaxPooling2D((3,3),strides=2))
	model.add(Flatten())
	model.add(Dense(64))#Shoudl be 128, I'm adjusting for lack of multiresolution
	model.add(Dense(1, activation='softmax'))
	return model

# final netowrk in the cascade. 
def make_48net(window_size = 48):
	model = Sequential()
	model.add(Conv2D(64,(5,5),strides=1,input_shape=(window_size,window_size,3)))
	model.add(MaxPooling2D((3,3),strides=2))
	model.add(BatchNormalization())
	model.add(Conv2D(64,(5,5),strides=1))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((3,3),strides=2))
	model.add(Dense(128), activation='softmax')#Shoudl be 256, I'm adjusting for lack of multiresolution
	return model



def cvprLi_network(img):
	net12 = make_12net()
	net24 = make_24net()
	net48 = make_48net()

	#for x,y,win in sliding_window_generator(12):
		#get predictiosn from 12 net

	#run predictions from 12 net on 24 net 
