import keras
from keras.layers import Conv2D, Dense, MaxPool2D, Input, PReLU, Reshape
from keras.optimizers import adam
# copied from https://github.com/xiangrufan/keras-mtcnn/blob/master/training/keras_12net_v1.py 
def pnet():
	inputs = Input(shape = [12,12,3]) # change this shape to [None,None,3] to enable arbitraty shape input
	x = Conv2D(10,(3,3),strides=1,padding='valid',name='conv1')(inputs)
	x = PReLU(shared_axes=[1,2],name='prelu1')(x)
	x = MaxPool2D(pool_size=2)(x) 
	x = Conv2D(16,(3,3),strides=1,padding='valid',name='conv2')(x)
	x = PReLU(shared_axes=[1,2],name='prelu2')(x)
	x = Conv2D(32,(3,3),strides=1,padding='valid',name='conv3')(x)
	x = PReLU(shared_axes=[1,2],name='prelu3')(x)
	classifier = Conv2D(2, (1, 1), activation='softmax',name='classifier1')(x)
	#classify = Reshape((2,))(classify) # look into this
	#regression = Conv2D(4, (1, 1),name='bbox1')(x) # aligning the box closer to the face. 
	#regression = Reshape((4,))(regression)
	return inputs, classifier
