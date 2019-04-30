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
from sklearn.preprocessing import normalize, scale
pd.set_option('display.max_columns', 10)
_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root


BATCH_SIZE = 256
EPOCHS = 500
do_train_model = True

_net_size = 24 # use this to change network.
IM_WIDTH=_net_size # legacy variables
IM_HEIGHT=_net_size

train_path = os.path.join(_root_project_dir,r"data/{}px_30k/train_db.pkl".format(IM_HEIGHT))
val_path = os.path.join(_root_project_dir,r"data/{}px_30k/val_db.pkl".format(IM_HEIGHT))
#test_path = os.path.join(_root_project_dir,r"data/{}px_reg/".format(IM_HEIGHT))

# Loads images into memory from a DB pickle file. pkl_file is the path to db file. Returns each collumn individually. 
def load_dataset(pkl_file):
  with open(pkl_file,'rb') as f:
    db = pickle.load(f)
    df = pd.DataFrame.from_records(db, columns=["path","class","box"])
    df = df.sample(frac=1).reset_index(drop=True) # shuffle the database
    ims = []
    for i,row in df.iterrows():
        this_im = cv2.imread(os.path.join(_root_project_dir,row['path']))
        norm_im = cv2.normalize(this_im, None, alpha=-1, beta=+1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ims.append(norm_im)
    
    classes = df.iloc[:,1].values
    bboxes = df.iloc[:,2].values
    bboxes = [x for x in bboxes] # flattens
    return np.asarray(ims), np.asarray(classes), np.asarray(bboxes)

print("Getting training data")
im_train, cls_train, reg_train = load_dataset(train_path)
print("Got ",len(im_train)," images for training.")
print("GEtting val data")
im_val, cls_val, reg_val = load_dataset(val_path)
print("im_train shape is ",im_train.shape, " val is", im_val.shape)
# data augmentation
#for i,im in enumerate(im_train):
#    x1,y1,w,h = reg_train[i]
#    im = np.fliplr(im)
#    n_x1 = IM_WIDTH-x1-w
#    reg_train[i][0] = n_x1 # Change coord of x1


# One hot encoding class labels
cls_train = to_categorical(cls_train)
cls_val = to_categorical(cls_val)

inputs, classif, regr = networks.mtcnn_rnet() if _net_size==24  else networks.mtcnn_pnet()
model = Model()
model = Model([inputs], [classif, regr])

losses = { "class_output":"categorical_crossentropy", "regr_output":"mse" }
lossWeights = {"class_output": 1.0, "regr_output": 1.0}
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
        ax[i].set_ylim([0,30])
        title = "Combined loss"
    if l == "class_output_loss": 
        ax[i].set_ylim([0,1])
        title = "Classification loss"
    if l == "regr_output_loss": 
        ax[i].set_ylim([0,30])
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
plt.savefig("{}_net_batch_size_{}30kdb.png".format(IM_WIDTH, BATCH_SIZE))
plt.close()
