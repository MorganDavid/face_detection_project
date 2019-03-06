import keras
from keras.layers import Conv2D, Dense, MaxPool2D, Input, PReLU, Reshape, Flatten
from keras.optimizers import adam

def pnet():
	inputs = Input(shape=(12,12,3))
	x = Conv2D(10,(3,3),strides=1)(inputs)
	x = PReLU(shared_axes=[1,2],)(x)
	x = MaxPool2D(pool_size=2)(x)
	x = Conv2D(16,(3,3),strides=1)(x)
	x = PReLU(shared_axes=[1,2])(x)
	x = Conv2D(32,(3,3),strides=1)(x)
	x = PReLU(shared_axes=[1,2])(x)
	x = PReLU(shared_axes=[1,2])(x)
	x = Flatten()(x)
	x = Dense(16)(x)
	predictions = Dense(1,activation='softmax')(x)
	return inputs,predictions
	