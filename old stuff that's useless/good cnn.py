import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
import numpy
import os
import pandas as pd
import cv2 
from os.path import dirname, abspath


_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root

train_path = os.path.join(_root_project_dir,r"data/20px/train")
val_path = os.path.join(_root_project_dir,r"data/20px/validation")
test_path = os.path.join(_root_project_dir,r"data/20px/test")

BATCH_SIZE = 20
do_train_model = True

IM_WIDTH=20
IM_HEIGHT=20

train_datagen = ImageDataGenerator(horizontal_flip=True,rotation_range=10,width_shift_range=3,height_shift_range=3,zoom_range=0.12)
test_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator(horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        directory=train_path,  # this is the target directory
        target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb')

validation_generator = validation_datagen.flow_from_directory(
        directory=val_path,  # this is the target directory
        target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb')

test_generator = test_datagen.flow_from_directory(
      directory=test_path,  # this is the target directory
      target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
      batch_size=BATCH_SIZE,
      class_mode='binary',
      color_mode='rgb')

#model architecture
def makeModel():
  model = keras.models.Sequential()
  model.add(Conv2D(16, (3, 3), input_shape=(IM_WIDTH, IM_HEIGHT,3),data_format='channels_last',activation='relu'))
  model.add(Dropout(0.2))
  model.add(Conv2D(16, (3, 3),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
  
  #model.add(Conv2D(64, (3, 3),activation='relu'))
  #model.add(Dropout(0.4))
  #model.add(Conv2D(32, (3, 3),activation='relu'))
  #model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

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
      epochs=18,
      validation_data=validation_generator,
      validation_steps=40,
      class_weight='auto')
  model.save('test_model.h5')

  predictions = model.evaluate_generator(test_generator)
  print(predictions)
  print(model.metrics_names)
else:
  model = load_model('test_model.h5')
  live_datagen = ImageDataGenerator()
  test_generator = test_datagen.flow_from_directory(
      directory=test_path,  # this is the target directory
      target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
      batch_size=BATCH_SIZE,
      class_mode='binary',
      color_mode='rgb')
  predictions = model.predict_generator(test_generator)
  print(predictions)
  print(model.evaluate_generator(test_generator))


