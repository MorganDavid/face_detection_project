import keras
from keras.layers import Conv2D, Dense, MaxPool2D, Input, ReLU, Reshape, Flatten, Dropout, Lambda
from keras.optimizers import adam
from keras.backend import batch_flatten
IM_HEIGHT=12#Make sure these are the same in train_network as they are here!
IM_WIDTH=12

'''
My Network
85% acc. Very poor speed. 
'''
def my_full_net():
  model = keras.models.Sequential()
  model.add(Conv2D(16, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH,3),data_format='channels_last',activation='relu'))
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
  model.add(Dense(1,activation='sigmoid'))

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
	inputs = Input(shape = [12,12,3]) # change this shape to [None,None,3] to enable arbitraty shape input
	x = Conv2D(10,(3,3),strides=1,padding='valid',name='conv1')(inputs)
	x = PReLU(shared_axes=[1,2],name='prelu1')(x)
	x = MaxPool2D(pool_size=2)(x) 
	x = Conv2D(16,(3,3),strides=1,padding='valid',name='conv2')(x)
	x = PReLU(shared_axes=[1,2],name='prelu2')(x)
	x = Conv2D(32,(3,3),strides=1,padding='valid',name='conv3')(x)
	x = PReLU(shared_axes=[1,2],name='prelu3')(x)
	classifier = Conv2D(2, (1, 1), activation='softmax',name='classifier1')(x) # construct 
	classify = Reshape((2,))(classify) # Flatten the coords
	regression = Conv2D(4, (1, 1),name='bbox1')(x) # aligning the box closer to the face. 
	regression = Reshape((4,))(regression)
	return inputs, classifier
