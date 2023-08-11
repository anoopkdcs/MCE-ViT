#!/usr/bin/env python
# coding: utf-8
#@author: Manjary & Anoop 

#### ViT Fine-tuning for DIF 

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
from PIL import Image, UnidentifiedImageError
from requests.exceptions import HTTPError
from io import BytesIO
from pathlib import Path
import torch
import pytorch_lightning as pl
from huggingface_hub import HfApi, HfFolder, Repository, notebook_login
from torch.utils.data import DataLoader
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

# Read dataset path
train_data_dir = Path("path to file\\train\\")
validation_data_dir = Path('path to file\\val\\')
#test_data_dir = Path('path to file\\test\\')

### ead data from folder 
ds = ImageFolder(train_data_dir)
val_ds = ImageFolder(validation_data_dir)
#test_ds = ImageFolder(test_data_dir)

print("Train data count: " +str(len(ds)))
print("Val data count: " +str(len(val_ds)))
#print("Test data count: " +str(len(test_ds)))


### Print samples from datast 
plt.figure(figsize=(20,10))
num_examples_per_class = 5
i = 1
for class_idx, class_name in enumerate(ds.classes):
    folder = ds.root / class_name
    print(folder)
    
    for image_idx, image_path in enumerate(sorted(folder.glob('*'))):
        if image_path.suffix in ds.extensions:
            #print("anoop")
            image = Image.open(image_path)
            plt.subplot(len(ds.classes), num_examples_per_class, i)
            ax = plt.gca()
            ax.set_title(class_name, size='xx-large', pad=5, loc='left', y=0, backgroundcolor='white')
            ax.axis('off')
            plt.imshow(image)
            i += 1
            if image_idx + 1 == num_examples_per_class:
                break

### Preparing Dataset Labels
label2id = {}
id2label = {}

for i, class_name in enumerate(ds.classes):
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
# 3. google/vit-huge-patch14-224-in21k
pre_trained_model = 'google/vit-large-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(pre_trained_model)

model = ViTForImageClassification.from_pretrained(
    pre_trained_model,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

collator = ImageClassificationCollator(feature_extractor)

batch_size = 16
train_loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator, shuffle=True) #num_workers=4, 
val_loader = DataLoader(val_ds, collate_fn=collator,  batch_size=batch_size) # num_workers=4
#test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collator) #, num_workers=4

### Training Model Class 
class Classifier(pl.LightningModule):

    def __init__(self, model, lr: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
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
        pred = outputs.logits.argmax(1)
        #print("prediction:"+ str(pred)+ "baatch ID: " + str(batch_idx))
        return pred
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

### seting-up the model to train
model_path = 'path to file\\models\\'

pl.seed_everything(42)
classifier = Classifier(model, lr=2e-5)
checkpoint_callback = ModelCheckpoint(dirpath = model_path,
                                      monitor = 'val_loss', save_top_k = 1, mode='min', verbose=True)
trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, max_epochs=25, callbacks=[checkpoint_callback]) #, strategy = 'dp'  

### finetuning the model 
trainer.fit(classifier, train_loader, val_loader)
test_data_dir = Path('path to file\\test\\')
test_ds = ImageFolder(test_data_dir)
print("Test data count: " +str(len(test_ds)))
test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collator) #, num_workers=4

### Geting Accuracy for test Split 
print("Test Accuracy and Loss")
trainer.test(classifier,test_loader)

### Geting Accuracy for Validation Split 
print("Validation Accuracy and Loss")
trainer.test(classifier,val_loader)

### Geting Accuracy for Validation Split 
print("Train Accuracy and Loss")
trainer.test(classifier,train_loader)
