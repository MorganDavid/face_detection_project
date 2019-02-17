import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import os
import pandas as pd
import cv2 

FACES_PATH="Faces/"
INVALID_PATH="Invalid/"
BATCH_SIZE = 20
do_train_model = False # set this to false if you just want to input weights.

IM_WIDTH=45
IM_HEIGHT=45
MODEL_FILENAME = "model.h5"

if __name__ == "__main__":
  train_datagen = ImageDataGenerator(horizontal_flip=True,rotation_range=10,width_shift_range=3,height_shift_range=3,zoom_range=0.12,rescale=1./255)
  test_datagen = ImageDataGenerator(rescale=1./255)
  validation_datagen = ImageDataGenerator(horizontal_flip=True,rescale=1./255)
  train_generator = train_datagen.flow_from_directory(
          directory=r"datasets/training_set_1000",  
          target_size=(IM_WIDTH, IM_HEIGHT),  
          batch_size=BATCH_SIZE,
          class_mode='binary',
          color_mode='rgb')

  validation_generator = validation_datagen.flow_from_directory(
          directory=r"datasets/validation_set_100",  
          target_size=(IM_WIDTH, IM_HEIGHT), 
          batch_size=BATCH_SIZE,
          class_mode='binary',
          color_mode='rgb')

  test_generator = test_datagen.flow_from_directory(
        directory=r"datasets/test_set_LFW",
        target_size=(IM_WIDTH, IM_HEIGHT),  
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False)



#model architecture
def makeModel():
  model = keras.models.Sequential()
  model.add(Conv2D(16, (3, 3), input_shape=(IM_WIDTH,IM_HEIGHT,3),data_format='channels_last',activation='relu', padding='same'))
  model.add(Conv2D(16, (3, 3),activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (5, 5),activation='relu', padding='same'))
  model.add(Conv2D(64, (5, 5),activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (5, 5),activation='relu', padding='same'))
  model.add(Conv2D(64, (5, 5),activation='relu', padding='same'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(32,activation='relu'))
  model.add(Dense(64,activation='relu'))
  model.add(Dense(64,activation='relu'))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  opt = keras.optimizers.SGD(decay=50e-9,momentum=0.9)

  model.compile(loss='binary_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
  return model
if __name__ == "__main__":#this means the model will only start fitting if we are running this class.
  if do_train_model:
    model = makeModel()
    print("RUNNING FIT GENERATOR NEXT")
    #model.load_weights("vgg_face_weights.h5")
    model.fit_generator(
        train_generator,
        steps_per_epoch=256,
        epochs=12,
        validation_data=validation_generator,
        validation_steps=40,
        class_weight='auto')
   
    model.save("model.h5")
    del model
   
  model = load_model(MODEL_FILENAME)

  evaluation = model.evaluate_generator(test_generator)
  print(evaluation)
  print(model.metrics_names)


  preds = model.predict_generator(test_generator)
  preds_flattened = [i>0.5 for [i] in preds]#False in these results means it IS a face. 
  #print(test_generator.classes)
  print(preds_flattened)
  print(confusion_matrix(preds_flattened,test_generator.classes))
  cl_names = ["face","not face"]
  print(classification_report(test_generator.classes, preds_flattened, target_names=cl_names))




