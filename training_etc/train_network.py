import keras
import numpy
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.models import Model
import numpy
import os
from os.path import dirname, abspath
import pandas as pd
import cv2 
from keras.optimizers import adam
import cvpr_networks as networks

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root

train_path = os.path.join(_root_project_dir,r"data/train")
val_path = os.path.join(_root_project_dir,r"data/validation")
test_path = os.path.join(_root_project_dir,r"data/test")



BATCH_SIZE = 20
do_train_model = True

IM_WIDTH=12
IM_HEIGHT=12

train_datagen = ImageDataGenerator(horizontal_flip=True,rotation_range=10,width_shift_range=3,height_shift_range=3,zoom_range=0.12,data_format='channels_last')
test_datagen = ImageDataGenerator(data_format='channels_last')
validation_datagen = ImageDataGenerator(horizontal_flip=True,data_format='channels_last')
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
inputs, predictions = networks.pnet()

model = Model(inputs=[inputs], outputs=[predictions])
model.load_weights('12net.h5',by_name=True)
if do_train_model:
  myadam = adam(lr=0.00000001)
  model.compile(loss='mse',
                optimizer=myadam,
                metrics=['accuracy'])

  model.fit_generator(
      train_generator,
      steps_per_epoch=128,
      epochs=4000,
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


