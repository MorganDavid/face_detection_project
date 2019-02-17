import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import numpy
import os
import pandas as pd
import cv2 



FACES_PATH="Faces/"
INVALID_PATH="Invalid/"
BATCH_SIZE = 20
do_train_model = True

IM_WIDTH=20
IM_HEIGHT=20

train_datagen = ImageDataGenerator(horizontal_flip=True,rotation_range=10,width_shift_range=3,height_shift_range=3,zoom_range=0.12)
test_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator(horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        directory=r"datasets/training_set_1000",  # this is the target directory
        target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale')

validation_generator = validation_datagen.flow_from_directory(
        directory=r"datasets/validation_set_100",  # this is the target directory
        target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale')

test_generator = test_datagen.flow_from_directory(
      directory=r"datasets/test_set_LFW",  # this is the target directory
      target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
      batch_size=BATCH_SIZE,
      class_mode='binary',
      color_mode='grayscale')



#model architecture
def makeModel():
  model = keras.models.Sequential()
  model.add(Conv2D(16, (3, 3), input_shape=(IM_WIDTH, IM_HEIGHT,1),data_format='channels_last',activation='relu'))
  model.add(Dropout(0.2))
  model.add(Conv2D(16, (3, 3),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
  
  model.add(Conv2D(64, (3, 3),activation='relu'))
  model.add(Dropout(0.4))
  model.add(Conv2D(32, (3, 3),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=2))


  model.add(Flatten())
  model.add(Dense(32,activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(16,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  return model
model = makeModel()
if do_train_model:
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  model.fit_generator(
      train_generator,
      steps_per_epoch=128,
      epochs=25,
      validation_data=validation_generator,
      validation_steps=40,
      class_weight='auto')
  model.save_weights('test_model.h5')

  predictions = model.evaluate_generator(test_generator,1000)
  print(predictions)
  print(model.metrics_names)
else:
  model.load_weights('test_model.h5',by_name=True)
  live_datagen = ImageDataGenerator()
  test_generator = test_datagen.flow_from_directory(
      directory=r"live_images",  # this is the target directory
      target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
      batch_size=BATCH_SIZE,
      class_mode='binary',
      color_mode='rgb')
  predictions = model.predict_generator(test_generator)
  print(predictions)


