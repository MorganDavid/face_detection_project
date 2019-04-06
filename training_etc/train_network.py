import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model, Model
from keras.optimizers import adam
import numpy
import os
import pandas as pd
import cv2 
from os.path import dirname, abspath
import networks

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root

train_path = os.path.join(_root_project_dir,r"data/20px/train")
val_path = os.path.join(_root_project_dir,r"data/20px/validation")
test_path = os.path.join(_root_project_dir,r"data/20px/test")

BATCH_SIZE = 20
do_train_model = True

IM_WIDTH=12
IM_HEIGHT=12

train_datagen = ImageDataGenerator(horizontal_flip=True)#,rotation_range=10,width_shift_range=3,height_shift_range=3,zoom_range=0.12
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



inputs, preds = networks.cvpr_pnet()
model = Model(inputs, preds)
madam = adam(lr = 10e-6);
if do_train_model:
  model.compile(loss='mse',
                optimizer=madam,
                metrics=['accuracy'])

  model.fit_generator(
      train_generator,
      steps_per_epoch=64,
      epochs=85,
      validation_data=validation_generator,
      validation_steps=40,
      class_weight='auto')
  model.save('cvpr_pnet.h5')

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


