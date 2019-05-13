import keras
from keras.layers import Conv2D, Dense, MaxPool2D, Input, ReLU,PReLU, Reshape, Flatten, Dropout, Lambda, BatchNormalization
from keras.optimizers import adam
from keras.backend import batch_flatten
IM_HEIGHT=24#Make sure these are the same in train_network as they are here!
IM_WIDTH=24

'''
My Network
85% acc. Very poor speed. 
'''
def my_full_net():
  model = keras.models.Sequential()
  model.add(Conv2D(16, (3, 3), input_shape=(24, 24,3),data_format='channels_last',activation='relu'))
  model.add(Dropout(0.2))
  model.add(Conv2D(16, (3, 3),activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2),strides=2))
  
  #model.add(Conv2D(64, (3, 3),activation='relu'))
  #model.add(Dropout(0.4))

  model.add(Flatten())
  model.add(Dense(32,activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(16,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2,activation='softmax'))

  return model

'''
Networks from CVPR 2015 paper. 
'''
def cvpr_12net():
	inputs = Input(shape=(12,12,3))
	x = Conv2D(10,(3,3),strides=1)(inputs)
	x = MaxPool2D(pool_size=3,strides=2)(x)
	x = ReLU()(x)
	x = Flatten()(x)
	x = Dense(16)(x)
	x = ReLU()(x)
	predictions = Dense(1,activation='softmax')(x)
	return inputs,predictions

'''
Networks from MTCNN paper. 
'''
def mtcnn_pnet():
	inputs = Input(shape = [12,12,3])
	x = Conv2D(10,(3,3),strides=1)(inputs)
	x = PReLU(shared_axes=[1,2])(x) # share the weights through the depth of this convolution.
	x = MaxPool2D(pool_size=2)(x) 

	x = Conv2D(16,(3,3),strides=1)(x)
	x = PReLU(shared_axes=[1,2])(x)
	x = Conv2D(32,(3,3),strides=1)(x)
	x = PReLU(shared_axes=[1,2])(x)
	classifier = Conv2D(2, (1, 1), activation='softmax')(x) # classifier as 1 or 0
	classifier = Reshape((2,),name="class_output")(classifier) # flatten the classification output
	regression = Conv2D(4, (1, 1))(x) # outputs 4 coords of where face is in 12x12.
	regression = Reshape((4,),name="regr_output")(regression) 
	return inputs, classifier, regression

def cnn_48net():
	model = keras.models.Sequential()
	model.add(Conv2D(32,(3,3),strides=1,input_shape=[24,24,3]))
	model.add(PReLU(shared_axes=[1,2]))
	model.add(MaxPool2D(pool_size=3,strides=2,padding='same'))
	model.add(Conv2D(64,(3,3),strides=1,padding='same'))
	model.add(PReLU(shared_axes=[1,2]))
	model.add(MaxPool2D(pool_size=3,strides=2,padding='same'))
	model.add(Conv2D(64,(3,3),strides=1,padding='same'))
	model.add(PReLU(shared_axes=[1,2]))
	model.add(Flatten())
	model.add(Dense(256, activation='relu') )
	model.add(Dense(1, activation='sigmoid', name="class_output"))
	return model

def cnn_48net_func():
	inputs = Input(24,24,3)
	x = Conv2D(32,(3,3),strides=1)(inputs)
	x = PReLU(shared_axes=[1,2])(x)
	x = MaxPool2D(pool_size=3,strides=2)(x)
	x = Conv2D(64,(3,3),strides=1)(x)
	x = PReLU(shared_axes=[1,2])(x)
	x = MaxPool2D(pool_size=3,strides=2)(x)
	x = Conv2D(64,(3,3),strides=1)(x)
	x = PReLU(shared_axes=[1,2])(x)
	x = Flatten()(x)
	x = Dense(256, activation='relu') (x)
	classifier = Dense(2, activation='softmax', name="class_output")(x)
	return inputs, classifer
def mtcnn_rnet():
	inputs = Input(shape = [24,24,3])
	x = Conv2D(28,(3,3),strides=1,padding='same')(inputs)
	x = PReLU()(x)
	x = MaxPool2D(pool_size=3)(x) 

	x = Conv2D(48,(3,3),strides=1,padding='same')(x)
	x = BatchNormalization()(x)
	x = PReLU()(x)
	x = MaxPool2D(pool_size=3)(x) 

	x = Conv2D(64,(2,2),strides=1,padding='same')(x)
	x = PReLU()(x)

	x = Flatten()(x)

	x = Dense(128)(x)
	x = PReLU(shared_axes=[1])(x)
	classifier = Dense(2, name="class_output", activation='softmax')(x)
	regression = Dense(4, name="regr_output")(x)
	return inputs, classifier, regression

def mtcnn_onet():
	inputs = Input(shape = [48,48,3])
	x = Conv2D(32,(3,3),strides=1,padding='same')(inputs)
	x = BatchNormalization()(x)
	x = PReLU()(x)
	x = MaxPool2D(pool_size=3)(x) 

	x = Conv2D(64,(3,3),strides=1,padding='same')(x)
	x = BatchNormalization()(x)
	x = PReLU()(x)
	x = MaxPool2D(pool_size=3)(x) 

	x = Conv2D(128,(2,2),strides=1,padding='same')(x)
	x = BatchNormalization()(x)
	x = PReLU()(x)

	x = Flatten()(x)

	x = Dense(256)(x)
	x = PReLU(shared_axes=[1])(x)
	classifier = Dense(2, name="class_output", activation='softmax')(x)
	regression = Dense(4, name="regr_output")(x)
	return inputs, classifier, regression