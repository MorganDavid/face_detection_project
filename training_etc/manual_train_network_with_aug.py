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
from sklearn.preprocessing import normalize, scale
from sklearn import metrics
import sys
pd.set_option('display.max_columns', 10)
np.set_printoptions(threshold=sys.maxsize)

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root


BATCH_SIZE = 256
EPOCHS = 300
do_train_model = True
_model_out_name = 'R-net-new.h5'
_net_size = 24 # use this to change network.
IM_WIDTH=_net_size # legacy variables
IM_HEIGHT=_net_size

train_path = os.path.join(_root_project_dir,r"data/{}px_mined/train_db.pkl".format(IM_HEIGHT))
val_path = os.path.join(_root_project_dir,r"data/{}px_mined/val_db.pkl".format(IM_HEIGHT))
#test_path = os.path.join(_root_project_dir,r"data/{}px_30k/test_db.pkl".format(IM_HEIGHT))

# Loads images into memory from a DB pickle file. pkl_file is the path to db file. Returns each column individually. 
def load_dataset(pkl_file):
  with open(pkl_file,'rb') as f:
    db = pickle.load(f)
    df = pd.DataFrame.from_records(db, columns=["path","class","box"])
    df = df.sample(frac=1).reset_index(drop=True) # shuffle the database
    ims = []
    
    for i,row in df.iterrows():
        this_im = cv2.imread(os.path.join(_root_project_dir,row['path']))
        print(os.path.join(_root_project_dir,row['path']))
        if (this_im.shape[0] != _net_size): this_im = cv2.resize(this_im,(_net_size,_net_size),interpolation=cv2.INTER_LINEAR)
        norm_im = cv2.normalize(this_im, None, alpha=-1, beta=+1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ims.append(norm_im)
    
    classes = df.iloc[:,1].values
    bboxes = df.iloc[:,2].values
    bboxes = [x for x in bboxes] # flattens
    return np.asarray(ims), np.asarray(classes), np.asarray(bboxes)



if do_train_model:
    print("Getting data")
    im_train, cls_train, reg_train = load_dataset(train_path)
    print("Got ",len(im_train)," images for training. shape:",im_train.shape)
    im_val, cls_val, reg_val = load_dataset(val_path)
    print("Got ",len(im_val)," images for validation.")
#im_test, cls_test, reg_test = load_dataset(test_path)
#print("Got ",len(im_test)," images for testing.")

def my_norm(bboxes, do_clamp): # if do_clamp is true, anything <0 is 0 and anything >_net_size is 1
    bboxes = bboxes.astype(np.float)
    for i,box in enumerate(bboxes):
        for x,num in enumerate(box):
            bboxes[i][x] = float(min(max(0,num / _net_size),1) if do_clamp else num / _net_size)
    return bboxes

# One hot encoding class labels
if do_train_model:
    cls_train = to_categorical(cls_train)
    cls_val = to_categorical(cls_val)
    #reg_train = my_norm(reg_train,True)
    #reg_val = my_norm(reg_val,True)

#reg_test = my_norm(reg_test,True)
#cls_test = to_categorical(cls_test)

# Flips an image and bounding box. 
def flip_image(im, bbox):
    x1,y1,w,h = bbox
    im = np.fliplr(im)
    if max(x1,y1,w,h) > 0:
        n_x1 = IM_WIDTH-x1-w
        bbox[0] = n_x1 # Change coord of x1
    return (im,bbox)
width_shift_range = 8
rotation_range=10
# Manual batch generator to augment regression boxes with images. 
def batch_generator(ims, classes, regrs, batch_size):
    while True:
        inds = np.random.randint(0,ims.shape[0],size=batch_size)
        batch_ims = ims[inds]
        batch_classes = classes[inds]
        batch_regr = regrs[inds]

        for i in range(batch_ims.shape[0]):
            flip_roll = random.randint(0,10)
            shift_roll = random.randint(0,10)
            bright_roll = random.randint(0,10)

            this_im = batch_ims[i] # only GET values from these (I think)
            this_class = batch_classes[i]
            this_regr = batch_regr[i]

            if flip_roll == 10: # flip image
                batch_ims[i], batch_regr[i] = flip_image(this_im,this_regr)
            if shift_roll == 10: # horizontal shift
                amnt = random.randint(-width_shift_range,width_shift_range)
                batch_ims[i] = np.roll(this_im,amnt, axis=1)
                batch_regr[i][0] = this_regr[0]+amnt
            if shift_roll == 9: # vertical shift
                amnt = random.randint(-width_shift_range,width_shift_range)
                batch_ims[i] = np.roll(this_im,amnt, axis=0)
                batch_regr[i][1] = this_regr[1]+amnt
            if bright_roll == 10: # rotation
                rows,cols,_ = batch_ims[i].shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2),random.randint(-rotation_range,rotation_range),1)
                batch_ims[i] = cv2.warpAffine(batch_ims[i],M,(cols,rows))
                rem=i

       #disp_im = cv2.normalize(batch_ims[rem], None, 0, 255, cv2.NORM_MINMAX)
       #cv2.imshow("after",disp_im.astype(np.uint8))
       #cv2.waitKey()
       #cv2.destroyAllWindows()
        yield (batch_ims, {"class_output": batch_classes, "regr_output": batch_regr})


inputs, classif, regr = networks.mtcnn_rnet() if _net_size==24  else networks.mtcnn_pnet()
model = Model()
model = Model([inputs], [classif, regr])
model.summary()
losses = { "class_output":"categorical_crossentropy", "regr_output":"mse" }
lossWeights = {"class_output": 1.2, "regr_output": 1.0}
myadam = adam(lr=,,decay=1e-10/EPOCHS)
model.compile(optimizer=myadam, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
#callbck = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, mode='auto')
if do_train_model:
    H = model.fit_generator(batch_generator(im_train,cls_train,reg_train,BATCH_SIZE),
        validation_data=(im_val,
            {"class_output": cls_val, "regr_output": reg_val}),
        epochs=EPOCHS,
        #callbacks=[callbck],
        steps_per_epoch=BATCH_SIZE,
        verbose=1)

    model.save(_model_out_name)
    # plot the total loss, category loss, and color loss
    lossNames = ["loss", "class_output_loss", "regr_output_loss"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
    
    file_out = open("{}net-hardneg-{}.txt".format(_net_size,BATCH_SIZE),'w')

    # loop over the loss names
    # DISCLAIMER: A lot of the graph plotting code is taken from https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    for (i, l) in enumerate(lossNames):
        file_out.write("\n name:val_"+l+"\n"+(''.join([str(x) for x in list(zip([str(x) for x in np.arange(0, EPOCHS)],[str(x) for x in H.history["val_"+l]]))]))) # writes history to file for external graph

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
    print("saving")
    # save the losses figure and create a new figure for the accuracies
    fig.tight_layout()
    plt.savefig("{}-net_batch_size_batch_size-{}_forresults.png".format(_net_size, BATCH_SIZE))
    plt.close()
    file_out.close()
model = load_model(_model_out_name)
# test data
#print("test results")
#tst_res = model.evaluate(x=im_test,y={"class_output": cls_test, "regr_output": reg_test})
#print(model.metrics_names, ":", tst_res)
