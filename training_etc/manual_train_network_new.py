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

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root

'''
Changing between networks:
 - Change file dirs,
 - Change width and height,
 - Change network

'''
train_path = os.path.join(_root_project_dir,r"data/24px_reg/train_db.pkl")
val_path = os.path.join(_root_project_dir,r"data/24px_reg/val_db.pkl")
#test_path = os.path.join(_root_project_dir,r"data/12px_reg/")

BATCH_SIZE = 64
EPOCHS = 400
do_train_model = True

IM_WIDTH=24
IM_HEIGHT=24


# Loads images into memory from a DB pickle file. pkl_file is the path to db file. Returns each collumn individually. 
def load_dataset(pkl_file):
  with open(pkl_file,'rb') as f:
    db = pickle.load(f)
    df = pd.DataFrame.from_records(db, columns=["path","class","box"])
    df.sample(frac=1).reset_index(drop=True) # shuffle the database
    ims = []

    for i,row in df.iterrows():
      ims.append(cv2.imread(os.path.join(_root_project_dir,row['path'])))

    classes = df.iloc[:,1].values
    bboxes = df.iloc[:,2].values
    bboxes = [x for x in bboxes] # flattens.
    return np.asarray(ims), np.asarray(classes), np.asarray(bboxes)

print("Getting training data")
im_train, cls_train, reg_train = load_dataset(train_path)
print("Got ",len(im_train)," images for training.")
print("GEtting val data")
im_val, cls_val, reg_val = load_dataset(val_path)
print("Got",len(im_val), " images for val")
print(im_train.shape)
print(type(im_train))
# One hot encoding class labels
cls_train = to_categorical(cls_train)
cls_val = to_categorical(cls_val)

inputs, classif, regr = networks.mtcnn_rnet()
model = Model()
model = Model([inputs], [classif, regr])

losses = { "class_output":"categorical_crossentropy", "regr_output":"mse" }
lossWeights = {"class_output": 1.0, "regr_output": 5.0}
myadam = adam(lr=1e-3, decay=1e-9/EPOCHS)
model.compile(optimizer=myadam, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
#callbck = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, mode='auto')

H = model.fit(im_train,
    {"class_output": cls_train, "regr_output": reg_train},
    validation_data=(im_val,
        {"class_output": cls_val, "regr_output": reg_val}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    #callbacks=[callbck],
    verbose=1)
model.save('R_net_mtcnn.h5')
# plot the total loss, category loss, and color loss
lossNames = ["loss", "class_output_loss", "regr_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
# Disclaimer: Most of graph plotting code taken from https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation dat
    title = "unknown loss"
    if l == "loss": 
        ax[i].set_ylim([0,400])
        title = "Combined loss"
    if l == "class_output_loss": 
        ax[i].set_ylim([0,5])
        title = "Classification loss"
    if l == "regr_output_loss": 
        ax[i].set_ylim([0,110])
        title = "Bounding Box Regression loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
        label="val_" + l)
    ax[i].legend()
 
# save the losses figure and create a new figure for the accuracies
fig.tight_layout()
plt.savefig("{}_net_batch_size_{}5x_reg.png".format(IM_WIDTH, BATCH_SIZE))
plt.close()
