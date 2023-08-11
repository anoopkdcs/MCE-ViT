#!/usr/bin/env python
# coding: utf-8
#@author: Manjary & Anoop 

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

import torch
import tensorflow as tf

np.random.seed(7) # fix random seed for reproducibility

### Loading Test Feature Vectors

feature_vector_original_test = np.load('path to file /file name.npy')
feature_vector_processed_test = np.load('path to file/file name.npy')

image_names = np.load('path to fiel/file name.npy')

print("Shape of Feature Vector Original Test: " +str(feature_vector_original_test.shape))
print("Shape of Feature Vector Processed Test: " +str(feature_vector_processed_test.shape))
print("Shape of Image Names Test: " +str(image_names.shape))

original_fv_len = feature_vector_original_test.shape[1] - 1
processed_fv_len = feature_vector_processed_test.shape[1] - 1

print(original_fv_len)
print(processed_fv_len)


### Checking Label of original feature and processed feature

original_data_label_test = feature_vector_original_test[:,original_fv_len]
processed_data_label_test = feature_vector_processed_test[:,processed_fv_len]
print(all(original_data_label_test == processed_data_label_test))

fv_original_test_without_label = feature_vector_original_test[:,0:original_fv_len]
print("Shape of Feature vector without label (test): " +str(fv_original_test_without_label.shape))


'''
import skimage.measure

fv_original_test_pooled = skimage.measure.block_reduce(fv_original_test_without_label, (1,8), np.average)
print("Shape of Feature vector without label (test): " +str(fv_original_test_pooled.shape))
'''

avg_pool  = torch.nn.AvgPool1d(16) #, stride=2)

fv_original_test_pooled = avg_pool(torch.tensor(fv_original_test_without_label)).detach().numpy()
print("Shape of Feature vector without label (test): " +str(fv_original_test_pooled.shape))


### Final Test Dataset

test_data = np.concatenate((fv_original_test_pooled,feature_vector_processed_test),axis = 1)
print("Concatenated Train Data Shape" + str(test_data.shape))

total_fv_len = test_data.shape[1] - 1
total_fv_len

### X Y spliting

test_x = test_data[:,0:total_fv_len]
test_y = test_data[:,total_fv_len]
print("Shape of Test X: " + str(test_x.shape))
print("Shape of Test Y: " + str(test_y.shape))

### Label Encoding

label_encoder = LabelEncoder()
print("Test Label: "+ str(test_y[0:6]))
test_y_encoded = test_y
label_encoder.fit(test_y_encoded)
test_y_encoded = label_encoder.transform(test_y_encoded)
test_y_encoded = to_categorical(test_y_encoded)
print("Test Label Encoded: "+ str(test_y_encoded[0:6]))

### Load best model

model_path="path to file/file name.h5"
ann_model = load_model(model_path)

### model prediction

loss_test, acc_test = ann_model.evaluate(test_x, test_y_encoded, verbose=1)
print("Test Accuracy: " +str(acc_test))
print("Test Loss: " +str(loss_test))

predictions = np.argmax(ann_model.predict(test_x), axis = 1)
g_truth = test_y

### plot confusion matrix of the results

class_names=['gan', 'graphic', 'real']
cm = confusion_matrix(y_target=g_truth,
                      y_predicted=predictions,
                      binary=False)
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_normed=True,
                                cmap="YlGnBu",
                                colorbar=True)#,
                                #class_names=class_names)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names,rotation=0)
plt.yticks(tick_marks, class_names)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('top')

### Test classification report

test_samples = len(g_truth) #2400
print('Test Classification Report\n')
test_rpt = classification_report(g_truth, predictions, target_names=class_names)
print(test_rpt)

df = pd.DataFrame({'ground_truth':g_truth, 'predictions':predictions, 'file_name':image_names})
df.to_csv("path to file/file name.csv",index=False)
df

