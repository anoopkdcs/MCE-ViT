#!/usr/bin/env python
# coding: utf-8
#@author: Manjary & Anoop 

######### Install torch GPU version, pytorch-lightning and transformers ######### 
'''
Run this code segments only once to install the main packages such as torch and transformer
'''
# ! pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# ! pip install transformers pytorch-lightning --quiet


### Check torch version and GPU count ###
import torch
print(torch.__version__)
print(torch.cuda.device_count())


import requests
import math
import matplotlib.pyplot as plt
import shutil
from getpass import getpass
from PIL import Image, UnidentifiedImageError,ImageChops, ImageEnhance
from requests.exceptions import HTTPError
from io import BytesIO
from pathlib import Path
import torch
import pytorch_lightning as pl
from huggingface_hub import HfApi, HfFolder, Repository, notebook_login
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification
import os
from torchmetrics.classification import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import itertools
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#pip install -U albumentations
import albumentations as A
#from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import skimage
from skimage import color


#ref: https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-10-albumentations-dataloader/notebook
class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, total_classes=None):
        self.transform = transform
        self.data = []
        self.root = root_dir
        
        if total_classes:
            self.classnames  = os.listdir(root_dir)[:total_classes] # for test
        else:
            self.classnames = os.listdir(root_dir)
            
        for index, label in enumerate(self.classnames):
            root_image_name = os.path.join(root_dir, label)
            
            for i in os.listdir(root_image_name):
                full_path = os.path.join(root_image_name, i)
                self.data.append((full_path, index))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, target = self.data[index]
        img = np.array(Image.open(data))
        
        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations["image"]
        
        target = torch.from_numpy(np.array(target))
        img = torch.from_numpy(img)
        
        #print(type(img),img.shape, target)
        
        return img,target 


#Ref: https://www.kaggle.com/code/hirotaka0122/bengali-custom-albumentations-transforms
compression_quality = 90
tp = A.ImageCompression(quality_lower=compression_quality, quality_upper=compression_quality, always_apply=True, p=1)
 
def jpeg_diff(img):
        
    img = np.uint8(skimage.color.rgb2ycbcr(img))
    compr_img = tp(image=img)['image']
    diff = ImageChops.difference(Image.fromarray(img.astype(np.uint8)), Image.fromarray(compr_img.astype(np.uint8))) 
    diff = np.asarray(diff)
    diff[:,:,0] = np.zeros((np.shape(diff[:,:,0])))
    add = ImageChops.add(Image.fromarray(img.astype(np.uint8)),  Image.fromarray(diff.astype(np.uint8)))
    
    extrema = add.getextrema()
    max_diff_add = max([ex[1] for ex in extrema])
    if max_diff_add == 0:
        max_diff_add = 1
    scale = 255.0 / max_diff_add  
    add = ImageEnhance.Brightness(add).enhance(scale)
    return np.asarray(add)

class DIFF(ImageOnlyTransform):
    def __init__(self, always_apply: bool = True, p: float = 1):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets: Dict[str, str] = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params: Dict[Any, Any] = {}
        self.replay_mode = False
        self.applied_in_replay = False
        
    def apply(self, img, **params):
        return jpeg_diff(img)


# Read dataset path  

#train_data_dir = Path("path to file\\train\\")
#validation_data_dir = Path('path to file\\val\\')
#test_data_dir = Path('path to file\test\\')

data_dir = Path('path to file\\train') 


### Read original data from folder 
original_data = ImageFolder(data_dir, total_classes=3)
print("Original data count: " +str(len(original_data)))

image, label = original_data[0]
plt.imshow(image)
print("Data Label: " + str(label))

a_transform = A.Compose([DIFF()])

processed_data = ImageFolder(data_dir, total_classes=3, transform=a_transform)
print("Train data count: " +str(len(processed_data)))

image, label = processed_data[0]
plt.imshow(image)
print("Data Label: " + str(label))

### Preparing Dataset Labels
label2id = {}
id2label = {}
#test_data = original_data

for i, class_name in enumerate(original_data.classnames):
    label2id[class_name] = str(i)
    id2label[str(i)] = class_name

print("class and label details: "+str(id2label))

### Preparing Dataset Labels
label2id = {}
id2label = {}
#test_data = processed_data

for i, class_name in enumerate(processed_data.classnames):
    label2id[class_name] = str(i)
    id2label[str(i)] = class_name

print("class and label details: "+str(id2label))


### Image Classification Collator
'''
To apply our transforms to images, we'll use a custom collator class. 
We'll initialize it using an instance of ViTFeatureExtractor and pass the collator 
instance to torch.utils.data.DataLoader's collate_fn kwarg.
'''
class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
 
    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings 

# Init Feature Extractor, Model, Data Loaders
# For other models: https://huggingface.co/google/vit-large-patch16-224-in21k
# 1. google/vit-base-patch16-224-in21k
# 2. google/vit-large-patch16-224-in21k
pre_trained_model = 'google/vit-large-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(pre_trained_model)

model = ViTForImageClassification.from_pretrained(
    pre_trained_model,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    output_attentions = True, output_hidden_states = True
)

collator = ImageClassificationCollator(feature_extractor)

batch_size = 1
loader_original_data = DataLoader(original_data, batch_size=batch_size, collate_fn=collator, shuffle=False) 
loader_processed_data = DataLoader(processed_data, batch_size=batch_size, collate_fn=collator, shuffle=False)

print("Number of data points (original): " +str(len(loader_original_data.dataset)))
print("Number of data points (processed): " +str(len(loader_processed_data.dataset)))

fig = plt.figure(figsize=(10, 10))
for i in range(8):
  image, label = original_data[i]
  fig.add_subplot(4, 4, i+1)
  plt.imshow(image)

fig = plt.figure(figsize=(10, 10))
for i in range(8):
  image, label = processed_data[i]
  fig.add_subplot(4, 4, i+1)
  plt.imshow(image)

### Training Model Class 
class Classifier(pl.LightningModule):

    def __init__(self, model, lr: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        #self.save_hyperparameters(logger=False)
        self.model = model
        self.forward = self.model.forward
        self.val_acc = Accuracy(
            task='multiclass' if model.config.num_labels > 2 else 'binary',
            num_classes=model.config.num_labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"train_loss", outputs.loss)
        return outputs.loss 

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"val_loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"val_acc", acc, prog_bar=True)
        return outputs.loss 

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"Loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"Acc", acc, prog_bar=True)
        return outputs.loss
    
    def predict_step(self, batch, batch_idx):
        outputs = self(**batch)
        #print(outputs.hidden_states[12])
        pred = outputs.logits.argmax(1)
        #print("prediction:"+ str(pred)+ "baatch ID: " + str(batch_idx))
        return pred
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

pl.seed_everything(42)

chkpt_original = 'path to file\\chkpoint_file_name.ckpt'

### Creating new model for Feature Extraction 
model_original = Classifier(model, lr=2e-5)
chkpt_data_original = torch.load(chkpt_original)
model_original.load_state_dict(chkpt_data_original['state_dict']) # which returns: <All keys matched successfully>

### Extracting features from Original Data
total_data_points = len(original_data)
feature_tmp_original = []

for i in range(total_data_points):
    image_tmp, label_tmp = original_data[i]
    #print(image_tmp)
    input_tmp_original = feature_extractor(images=image_tmp, return_tensors="pt")
    outputs_tmp_original = model_original(**input_tmp_original)
    feature_tensor_original  = outputs_tmp_original.hidden_states[-1][0][0] #all(outputs.hidden_states[12][0][0] == outputs.hidden_states[-1][0][0]) return True
    feature_numpy_original = feature_tensor_original.detach().numpy() #.reshape((1, feature_vector_tensor.shape[0]))
    feature_numpy_original = np.append(feature_numpy_original, np.int32(label_tmp.numpy()))
    feature_tmp_original.append(feature_numpy_original)
    feature_vector_original = np.array(feature_tmp_original)
    print("Data point: " +str(i))
    
print(feature_vector_original.shape)

### Save Feature Vector of Original Date  
np.save('path to file\\feature_file_name.npy', feature_vector_original) 
# feature_vector_original = np.load('path to fiel/feature_file_name.npy')


chkpt_processed = 'path to file\\checkpoint_file_name.ckpt'

### Creating new model for Feature Extraction 
model_processed = Classifier(model, lr=2e-5)
chkpt_data_processed = torch.load(chkpt_processed)
model_processed.load_state_dict(chkpt_data_processed['state_dict']) # which returns: <All keys matched successfully>


### Extracting features from Processed Data
total_data_points = len(processed_data)
feature_tmp_processed = []

for i in range(total_data_points):
    image_tmp, label_tmp = processed_data[i]
    #print(image_tmp)
    input_tmp_processed = feature_extractor(images=image_tmp, return_tensors="pt")
    outputs_tmp_processed = model_processed(**input_tmp_processed)
    feature_tensor_processed  = outputs_tmp_processed.hidden_states[-1][0][0] #all(outputs.hidden_states[12][0][0] == outputs.hidden_states[-1][0][0]) return True
    #print(feature_vector_tensor)
    feature_numpy_processed = feature_tensor_processed.detach().numpy() #.reshape((1, feature_vector_tensor.shape[0]))
    feature_numpy_processed = np.append(feature_numpy_processed, np.int32(label_tmp.numpy()))
    feature_tmp_processed.append(feature_numpy_processed)
    feature_vector_processed = np.array(feature_tmp_processed)
    print("Data point: " +str(i))

print(feature_vector_processed.shape)

### Save Feature Vector of Processed Date  
np.save('file to path\\feature_file_name.npy', feature_vector_processed) 
