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
import matplotlib.pyplot as plt

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root

train_path = os.path.join(_root_project_dir,r"data/24px_mined/train")
val_path = os.path.join(_root_project_dir,r"data/24px_mined/val")
test_path = os.path.join(_root_project_dir,r"data/24px_mined/test")

BATCH_SIZE = 200
do_train_model = True

IM_WIDTH=24
IM_HEIGHT=24

train_datagen = ImageDataGenerator(horizontal_flip=True,rescale=1./255)#,rotation_range=10,width_shift_range=3,height_shift_range=3,zoom_range=0.12
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(horizontal_flip=True,rescale=1./255)
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


# If we are using a functional method
#from keras.backend import batch_flatten
#inputs, preds = networks.cvpr_12net()
#model = Model(inputs, preds)

# if we are using a non-functional method:
model = networks.cnn_48net()
madam = adam(lr = 10e-6);
if do_train_model:
  model.compile(loss='binary_crossentropy',
                optimizer=madam,
                metrics=['accuracy'])

  history = model.fit_generator(
      train_generator,
      steps_per_epoch=64,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=40,
      class_weight='auto')
  model.save('test.h5')
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  predictions = model.evaluate_generator(test_generator)
  print(predictions)
  print(model.metrics_names)
else:
  model = load_model('test.h5')
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


