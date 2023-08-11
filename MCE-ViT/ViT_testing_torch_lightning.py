#!/usr/bin/env python
# coding: utf-8
#@author: Manjary & Anoop 

#### ViT testing for DIF 

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
test_data_dir = Path('path to file\\data')

### ead data from folder 
test_ds = ImageFolder(test_data_dir)

print("Test data count: " +str(len(test_ds)))

### Print samples from datast 
plt.figure(figsize=(20,10))
num_examples_per_class = 5
i = 1
for class_idx, class_name in enumerate(test_ds.classes):
    folder = test_ds.root / class_name
    for image_idx, image_path in enumerate(sorted(folder.glob('*'))):
        if image_path.suffix in test_ds.extensions:
            #print("anoop")
            image = Image.open(image_path)
            plt.subplot(len(test_ds.classes), num_examples_per_class, i)
            ax = plt.gca()
            ax.set_title(class_name, size='xx-large', pad=5, loc='left', y=0, backgroundcolor='white')
            ax.axis('off')
            plt.imshow(image)
            i += 1
            if image_idx + 1 == num_examples_per_class:
                break


# In[6]:


### Preparing Dataset Labels
label2id = {}
id2label = {}

for i, class_name in enumerate(test_ds.classes):
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
pre_trained_model = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(pre_trained_model)

model = ViTForImageClassification.from_pretrained(
    pre_trained_model,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

collator = ImageClassificationCollator(feature_extractor)

test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collator) #, num_workers=4


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


### Test from best checkpoint
pl.seed_everything(42)
classifier = Classifier(model, lr=2e-5)
chkpt = 'path to file\\checkpoint_file_name.ckpt'
trainer = pl.Trainer(accelerator="gpu", devices=1, )
predictions = trainer.predict(classifier,test_loader, ckpt_path = chkpt)

### Converting prediction tensor into LIST of predictions
predicted = []
data = predictions[:]
for i in range(len(data)):
  temp = data[i][0].tolist()
  predicted.append(temp)

print("Length of prediction LIST: " +str(len(predicted)))

### Creatinh ground-truth labels form test loader 
batches = iter(test_loader)
tmp_labels = []
for i in range(len(batches)):
  tmp_data = next(batches) 
  key, data = tmp_data.items()
  tmp_labels.append(data[1].tolist())

gt_labels = list(itertools.chain(*tmp_labels))
print("Length of ground-truth LIST: " +str(len(gt_labels)))


### plot confusion matrix of the results 

class_names=['gan', 'graphic', 'real']
cm = confusion_matrix(y_target=gt_labels, 
                      y_predicted=predicted, 
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
test_samples = len(gt_labels) #2400
print('\nTest Classification Report\n')
test_rpt = classification_report(gt_labels, predicted, target_names=class_names)
print(test_rpt)


### Geting Accuracy for test Split 
trainer.test(classifier,test_loader)





