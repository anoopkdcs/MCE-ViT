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


### Loading Feature Vectors 
np.random.seed(7) # fix random seed for reproducibility

feature_vector_original_train = np.load('path to file\\file_name.npy')
feature_vector_processed_train = np.load('path to file\\file_name.npy')
print("Shape of Feature Vector Original Train: " +str(feature_vector_original_train.shape))
print("Shape of Feature Vector Processed Train: " +str(feature_vector_processed_train.shape))

feature_vector_original_val = np.load('path to file\\file_name.npy')
feature_vector_processed_val = np.load('path to file\\file_name.npy')
print("\nShape of Feature Vector Original Validation: " +str(feature_vector_original_val.shape))
print("Shape of Feature Vector Processed Validation: " +str(feature_vector_processed_val.shape))

feature_vector_original_test = np.load('path to file name\\file_name.npy')
feature_vector_processed_test = np.load('path to file\\file_name.npy')
print("\nShape of Feature Vector Original Test: " +str(feature_vector_original_test.shape))
print("Shape of Feature Vector Processed Test: " +str(feature_vector_processed_test.shape))

original_fv_len = feature_vector_original_train.shape[1] - 1
processed_fv_len = feature_vector_processed_train.shape[1] - 1

print(original_fv_len)
print(processed_fv_len)

### Checking Label of original feature and processed feature
original_data_label_train = feature_vector_original_train[:,original_fv_len]
processed_data_label_train = feature_vector_processed_train[:,processed_fv_len]
print(all(original_data_label_train == processed_data_label_train))
print("Shape of Feature label Original Train: " +str(original_data_label_train.shape))
print("Shape of Feature label Processed Train: " +str(processed_data_label_train.shape))
print("")

original_data_label_val = feature_vector_original_val[:,original_fv_len]
processed_data_label_val = feature_vector_processed_val[:,processed_fv_len]
print(all(original_data_label_val == processed_data_label_val))
print("Shape of Feature label Original Validation: " +str(original_data_label_val.shape))
print("Shape of Feature label Processed Validation: " +str(processed_data_label_val.shape))
print("")

original_data_label_test = feature_vector_original_test[:,original_fv_len]
processed_data_label_test = feature_vector_processed_test[:,processed_fv_len]
print(all(original_data_label_test == processed_data_label_test))
print("Shape of Feature label Original Test: " +str(original_data_label_test.shape))
print("Shape of Feature label Processed Test: " +str(processed_data_label_test.shape))

fv_original_train_without_label = feature_vector_original_train[:,0:original_fv_len]
fv_original_val_without_label = feature_vector_original_val[:,0:original_fv_len]
fv_original_test_without_label = feature_vector_original_test[:,0:original_fv_len]
print("Shape of Feature vector without label (train): " +str(fv_original_train_without_label.shape))
print("Shape of Feature vector without label (val): " +str(fv_original_val_without_label.shape))
print("Shape of Feature vector without label (test): " +str(fv_original_test_without_label.shape))

'''
import skimage.measure

fv_original_train_pooled = skimage.measure.block_reduce(fv_original_train_without_label, (1,16), np.average)
fv_original_val_pooled = skimage.measure.block_reduce(fv_original_val_without_label, (1,16), np.average)
fv_original_test_pooled = skimage.measure.block_reduce(fv_original_test_without_label, (1,16), np.average)
print("Shape of Feature vector without label (train): " +str(fv_original_train_pooled.shape))
print("Shape of Feature vector without label (val): " +str(fv_original_val_pooled.shape))
print("Shape of Feature vector without label (test): " +str(fv_original_test_pooled.shape))
'''

import torch
import tensorflow as tf 

#avg_pool  = torch.nn.AvgPool1d(24) # , stride=2)
fv_original_train_pooled = fv_original_train_without_label # avg_pool(torch.tensor(fv_original_train_without_label)).detach().numpy()
fv_original_val_pooled = fv_original_val_without_label # avg_pool(torch.tensor(fv_original_val_without_label)).detach().numpy()
fv_original_test_pooled = fv_original_test_without_label #avg_pool(torch.tensor(fv_original_test_without_label)).detach().numpy()

print("Shape of Feature vector without label (train): " +str(fv_original_train_pooled.shape))
print("Shape of Feature vector without label (val): " +str(fv_original_val_pooled.shape))
print("Shape of Feature vector without label (test): " +str(fv_original_test_pooled.shape))


### Final Train, Val, and Test Dataset
train_data = np.concatenate((fv_original_train_pooled,feature_vector_processed_train),axis = 1)
np.random.shuffle(train_data) # for shuffle the train data
print("Concatenated train Data Shape" + str(train_data.shape))

val_data = np.concatenate((fv_original_val_pooled,feature_vector_processed_val),axis = 1)
#np.random.shuffle(train_data) # for shuffle the train data
print("\nConcatenated val Data Shape" + str(val_data.shape))

test_data = np.concatenate((fv_original_test_pooled,feature_vector_processed_test),axis = 1)
print("\nConcatenated test Data Shape" + str(test_data.shape))


total_fv_len = train_data.shape[1] - 1
total_fv_len

### X Y spliting 
train_x = train_data[:,0:total_fv_len]
train_y = train_data[:,total_fv_len]
print("Shape of Train X: " + str(train_x.shape))
print("Shape of Train Y: " + str(train_y.shape))

val_x = val_data[:,0:total_fv_len]
val_y = val_data[:,total_fv_len]
print("\nShape of Val X: " + str(val_x.shape))
print("Shape of Val Y: " + str(val_y.shape))

test_x = test_data[:,0:total_fv_len]
test_y = test_data[:,total_fv_len]
print("\nShape of Test X: " + str(test_x.shape))
print("Shape of Test Y: " + str(test_y.shape))

### Label Encoding 
label_encoder = LabelEncoder()
print("Train Label: "+ str(train_y[0:5]))
train_y_encoded = train_y
label_encoder.fit(train_y_encoded)
train_y_encoded = label_encoder.transform(train_y_encoded)
train_y_encoded = to_categorical(train_y_encoded)
print("Train Label Encoded: \n"+ str(train_y_encoded[0:5]))

print("\nVal Label: "+ str(val_y[0:5]))
val_y_encoded = val_y
label_encoder.fit(val_y_encoded)
val_y_encoded = label_encoder.transform(val_y_encoded)
val_y_encoded = to_categorical(val_y_encoded)
print("Val Label Encoded: \n"+ str(val_y_encoded[0:5]))

print("\nTest Label: "+ str(test_y[0:5]))
test_y_encoded = test_y
label_encoder.fit(test_y_encoded)
test_y_encoded = label_encoder.transform(test_y_encoded)
test_y_encoded = to_categorical(test_y_encoded)
print("Train Label Encoded: \n"+ str(test_y_encoded[0:5]))


input_size = train_x.shape[1]
model = Sequential()
model.add(Dense(512, input_dim=input_size, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()


filepath="path to file\\model_name.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_x, train_y_encoded, validation_data=(val_x, val_y_encoded), epochs=50, callbacks=callbacks_list, batch_size=16)

# list all data in history
#print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('path to file\\file_name.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('path to file\\file_name.png')
plt.show()

### load best model 
ann_model = load_model(filepath)


### Validate the results for Train Data, Validation Data, and Test Data

loss_train, acc_train = ann_model.evaluate(train_x, train_y_encoded, verbose=0)
print("Train Accuracy: " +str(acc_train))
print("Train Loss: " +str(loss_train))

loss_val, acc_val = ann_model.evaluate(val_x, val_y_encoded, verbose=0)
print("\nValidation Accuracy: " +str(acc_val))
print("Validation Loss: " +str(loss_val))

loss_test, acc_test = ann_model.evaluate(test_x, test_y_encoded, verbose=0)
print("\nTest Accuracy: " +str(acc_test))
print("Test Loss: " +str(loss_test))


from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report


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
print('\nTest Classification Report\n')
test_rpt = classification_report(g_truth, predictions, target_names=class_names)
print(test_rpt)

