import keras
import numpy

train_path = r"datasets/training_set_1000"
val_path = r"datasets/validation_set_100"
test_path = r"datasets/test_set_LFW"

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import numpy
import os
import pandas as pd
import cv2 

import networks

BATCH_SIZE = 20
do_train_model = True

IM_WIDTH=12
IM_HEIGHT=12

train_datagen = ImageDataGenerator(horizontal_flip=True,rotation_range=10,width_shift_range=3,height_shift_range=3,zoom_range=0.12)
test_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator(horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        directory=r"datasets/training_set_1000",  # this is the target directory
        target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb')

validation_generator = validation_datagen.flow_from_directory(
        directory=r"datasets/validation_set_100",  # this is the target directory
        target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb')

test_generator = test_datagen.flow_from_directory(
      directory=r"datasets/test_set_LFW",  # this is the target directory
      target_size=(IM_WIDTH, IM_HEIGHT),  # all images will be resized to 150x150
      batch_size=BATCH_SIZE,
      class_mode='binary',
      color_mode='rgb')



#model architecture
model = networks.make_12net()

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


