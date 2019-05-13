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
import keract
from inspect import signature
from matplotlib.patches import Rectangle 

_root_project_dir = dirname(dirname(dirname(abspath(__file__)))) # go up directories from where we are to get root
np.set_printoptions(threshold=sys.maxsize)

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
#cv2.imshow("t",cv2.normalize(ims[550], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8))
#cv2.waitKey()
#cv2.destroyAllWindows()
model = load_model('R-net-mining.h5')
ims=np.roll(ims,shift=1)
print(np.asarray(ims).shape)
#print(lbls)

y_preds = model.predict(x=ims)
y_preds = np.asarray(y_preds[0])
#print(y_preds)
y_preds_thresh = [1 if (x2-x1)>0.999 else 0 for x1,x2 in y_preds]
# Replace this in keract display_activations to make it accept batch images as input for the feature m ap display.
#for layer_name, acts in zip(['conv2d_1/BiasAdd:0','conv2d_3/BiasAdd:0'],[activations['conv2d_1/BiasAdd:0'], activations['conv2d_3/BiasAdd:0']]):#activations.items():
#    print("orignal shape",acts.shape)
#    acts = acts[0]
#    acts = np.asarray([acts])
#    print("shape is",acts.shape)
#    print(layer_name, acts.shape, end=' ')
#
#    if acts.shape[0] != 1:
#        print('-> Skipped. First dimension is not 1.')
#        continue
#    if len(acts.shape) <= 2:
#        print('-> Skipped. 2D Activations.')
#        continue


report = metrics.classification_report(y_true=y_preds_thresh, y_pred=np.asarray(lbls), output_dict =True)
print(report)
matrix = metrics.confusion_matrix(y_preds_thresh, np.asarray(lbls))
#print(matrix)
act= keract.get_activations(model,ims)
#print("keys",act.keys())
#keract.display_activations(act)
'''
print("act",act.keys())
one_act = np.reshape(act["conv2d_1/BiasAdd:0"][0],(28,24,24))
print("shape",one_act.shape)
print(one_act[0].shape)
cv2.imshow("t",one_act[0], cv2.IMREAD_GRAYSCALE)
cv2.waitKey()
'''
y_preds_roc = [(x2-x1) for x1,x2 in y_preds]
fpr, tpr, thresholds = metrics.roc_curve(lbls,y_preds_roc)
#print("AUC",metrics.roc_auc_score(lbls,y_preds_roc))

#plt.plot(fpr,tpr)
#plt.title("ROC Curve for R-net predictions on test set of 494.")
#plt.xlabel("False postive rate")
#plt.ylabel("True positive rate")
#plt.show()

# Disclaimer: fill plot code taken from sklearn documentation
average_precision = metrics.average_precision_score(lbls, y_preds_roc)

precision,recall,thr = metrics.precision_recall_curve(lbls,y_preds_roc)
print(precision.shape, " ", recall.shape, " ", thr.shape)
df = pd.DataFrame({"prec":precision[:-1],"rec":recall[:-1],"thsh":thr})
print(df.to_string())
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
fig,ax = plt.subplots()
ax.step(recall, precision, color='r', alpha=0.8,
         where='post')
ax.fill_between(recall,precision , alpha=0.15, color='r', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-recall for R-net')

ax.annotate('- Metrics -\nAverage precision:{:.2f}\nAverage Recall:{:.2f}'.format(average_precision,report['macro avg']['recall']), \
	xy=(0.015, 0.015), xycoords='data',
    size=10, ha='left', va='bottom',
    bbox=dict(boxstyle='round', fc='w'))


plt.show()