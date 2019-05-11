import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model, Model
from keras.optimizers import adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import os
import pandas as pd
import cv2 
from os.path import dirname, abspath
import networks
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
import sklearn
from sklearn.preprocessing import normalize, scale
from sklearn import metrics
import sys

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root

test_path = os.path.join(_root_project_dir,r"data/test")

ims = []
lbls = []

def load_dataset_no_pkl(dire):
  f = os.listdir(dire)
  for i,c in enumerate(f):
  	this_c = os.path.join(dire,c)
  	for filename in os.listdir(this_c):
  		im=cv2.imread(os.path.join(this_c,filename))
  		im = cv2.normalize(im, None, alpha=-1, beta=+1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  		im=cv2.resize(im,(24,24))
  		ims.append(im)
  		lbls.append(i)

load_dataset_no_pkl(test_path)
model = load_model('R-net_mining.h5')
ims=np.roll(ims,shift=1)
print(np.asarray(ims).shape)
print(lbls)

y_preds = model.predict(x=ims)
print(np.asarray(y_preds[0]).shape)
print("preds are ",y_preds[0])
y_preds = np.asarray(y_preds[0])
threashes = [0.6,0.7,0.8,0.85,0.9,0.99,0.999,0.9999]
out = [] # strucutre: [num_of_false_pos, threash, precison]
for thresh in threashes:
	y_preds_thresh = [1 if (x2-x1)>thresh else 0 for x1,x2 in y_preds]

	report = metrics.classification_report(y_true=y_preds_thresh, y_pred=np.asarray(lbls), output_dict =True)
	print(report)
	matrix = metrics.confusion_matrix(y_preds_thresh, np.asarray(lbls))
	print(matrix)
	out.append([matrix[1][1],report['0']['precision']])

print("out is ", out)
out = np.asarray(out)
print("false pos",out[:,0])

plt.plot(out[:,0],out[:,1])
plt.show()