#!/usr/bin/env python
# coding: utf-8
#@author: Manjary & Anoop 


######### Install torch GPU version, pytorch-lightning and transformers #########
'''
Run this code segments only once to install the main packages such as torch and transformer
'''
# for GPU machine
# ! pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# ! pip install transformers pytorch-lightning --quiet

'''
!pip install lightning
!pip install huggingface_hub
!pip install transformers
'''

### Refrence
# 1. https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map/notebook
# 2. https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb


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

# Read dataset path
data_dir = Path('path to file')

#ref: https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-10-albumentations-dataloader/notebook
class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, total_classes=None):
        self.transform = transform
        self.data = []
        self.root = root_dir

        if total_classes:
            self.classnames  = sorted(os.listdir(root_dir)[:total_classes]) # for test
        else:
            self.classnames = sorted(os.listdir(root_dir))

        for index, label in enumerate(self.classnames):
            root_image_name = os.path.join(root_dir, label)

            for i in sorted(os.listdir(root_image_name)):
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

original_input = ImageFolder(data_dir)
print("Test data count: " +str(len(original_input)))

image, label = original_input[0]
plt.imshow(image)
print(label)

### Preparing Dataset Labels
label2id = {}
id2label = {}

for i, class_name in enumerate(original_input.classnames):
    label2id[class_name] = str(i)
    id2label[str(i)] = class_name

print("class and label details: "+str(id2label))

#Ref: https://www.kaggle.com/code/hirotaka0122/bengali-custom-albumentations-transforms

compression_quality = 90
tp = A.ImageCompression(quality_lower=compression_quality, quality_upper=compression_quality, always_apply=True, p=1)

def jpeg_diff(img):
    img = np.uint8(skimage.color.rgb2ycbcr(img))
    compr_img = tp(image=img)['image']
    diff1 = ImageChops.difference(Image.fromarray(img.astype(np.uint8)), Image.fromarray(compr_img.astype(np.uint8)))
    diff1 = np.asarray(diff1)
    diff = diff1.copy()
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

a_transform = A.Compose(
    [
        DIFF()
    ]
)

processed_input = ImageFolder(data_dir,total_classes=3, transform=a_transform)
print("Test data count: " +str(len(processed_input)))

image, label = processed_input[0]
plt.imshow(image)
print(label)

### Preparing Dataset Labels
label2id = {}
id2label = {}

for i, class_name in enumerate(processed_input.classnames):
    label2id[class_name] = str(i)
    id2label[str(i)] = class_name

print("class and label details: "+str(id2label))

test_data = processed_input #original_input

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
    id2label=id2label, output_attentions = True
) #output_attentions = True will returen the attentions matrix also along with the predictions

collator = ImageClassificationCollator(feature_extractor)

test_loader = DataLoader(test_data, batch_size=1, collate_fn=collator) #, num_workers=4

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
chkpt_original = 'path to file/file name.ckpt'

### Creating new model for Feature Extraction
new_model = Classifier(model, lr=2e-5) #model_original
chkpt_data_original = torch.load(chkpt_original)
new_model.load_state_dict(chkpt_data_original['state_dict']) # which returns: <All keys matched successfully>

################ prediction from old model checkpoints

trainer = pl.Trainer(accelerator="gpu", devices=1,) #pl.Trainer(accelerator="cpu", devices=1,)
predictions = trainer.predict(new_model,test_loader, ckpt_path = chkpt_original)

### Converting prediction tensor into LIST of predictions
predicted = []
data = predictions[:]
for i in range(len(data)):
  temp = data[i][0].tolist()
  predicted.append(temp)

print("Length of prediction LIST: " +str(len(predicted)))


### Creating ground-truth labels form test loader
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
                                colorbar=True,
                                class_names=class_names)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names,rotation=0)
plt.yticks(tick_marks, class_names)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('top')


################ prediction from NEW model checkpoints

trainer = pl.Trainer(accelerator="gpu", devices=1, ) # pl.Trainer(accelerator="cpu", devices=1, )
predictions_new = trainer.predict(new_model,test_loader, ckpt_path = chkpt_original) # , ckpt_path = chkpt_original

### Converting prediction tensor into LIST of predictions
predicted = []
data = predictions_new[:]
for i in range(len(data)):
  temp = data[i][0].tolist()
  predicted.append(temp)

print("Length of prediction LIST: " +str(len(predicted)))


### Creating ground-truth labels form test loader
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
                                colorbar=True,
                                class_names=class_names)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names,rotation=0)
plt.yticks(tick_marks, class_names)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('top')


# **Attention Map VIsualization**

save_folder = 'path to file'

for image_number in range(0,20):

  print(image_number) #gan, graphics, real is the order

  original_image = original_input[image_number][0]
  processed_image = processed_input[image_number][0]

  attention_image = processed_image  #  original_image

  #fig1, (ax11, ax12) = plt.subplots(ncols=2, figsize=(10, 10))
  #ax11.set_title('Original Image')
  #ax12.set_title('Processed Image')
  #_ = ax11.imshow(original_image)
  #_ = ax12.imshow(processed_image)
  #plt.imsave(save_folder+str(image_number)+'_processed.jpg',np.uint8(processed_image))
  #cv2.imwrite(basepath_attn+filename+'_processed.jpg',np.uint8(processed_image))

  ### prediction
  extracted = feature_extractor(attention_image, return_tensors='pt')
  prediction = new_model(**extracted)
  class_intensity_prediction = prediction.logits
  attentions = prediction.attentions

  ### Extracting Attention matrix
  att_mat = torch.stack(attentions).squeeze(1)
  #print(att_mat.shape)

  # Average the attention weights across all heads.
  att_mat = torch.mean(att_mat, dim=1)
  #print(att_mat.shape)

  ### Attention map creation and embedding with input image

  # To account for residual connections, we add an identity matrix to the
  # attention matrix and re-normalize the weights.
  residual_att = torch.eye(att_mat.size(1))
  aug_att_mat = att_mat + residual_att
  aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

  # Recursively multiply the weight matrices
  joint_attentions = torch.zeros(aug_att_mat.size())
  joint_attentions[0] = aug_att_mat[0]

  for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

  # Attention from the output token to the input space.
  v = joint_attentions[-1]
  #print(v.shape)
  grid_size = int(np.sqrt(aug_att_mat.size(-1)))
  mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
  #print(image.shape)
  mask_result = cv2.resize(mask / mask.max(), (original_image.shape[0],original_image.shape[0]))
  mask = cv2.resize(mask / mask.max(), (original_image.shape[0],original_image.shape[0]))[..., np.newaxis]
  result = (mask * original_image.numpy()).astype("uint8")
  #print(mask.shape)
  #print(processed_image.shape)
  #print(type(mask))
  #print(type(processed_image))

  ### Plot the attention map embedded images
  #fig, (ax1, ax2,ax3) = plt.subplots(ncols=3, figsize=(10, 10))
  #ax1.set_title('Original')
  #ax2.set_title('Attention Map')
  #ax3.set_title('Attention Map')
  #_ = ax1.imshow(original_image)
  #_ = ax2.imshow(mask_result)
  #_ = ax3.imshow(result)
  plt.imsave(save_folder+str(image_number)+'_processed_attn.jpg', mask_result) #_original_attn
  #plt.imsave(save_folder+str(image_number)+'_processed_attn_on_im.jpg', np.uint8(result))

  probs = torch.nn.Softmax(dim=-1)(class_intensity_prediction)
  top5 = torch.argsort(probs, dim=-1, descending=True)
  #print(top5)
  #print("Prediction Label and Attention Map!\n")
  for idx in top5[0, :3]:
    #print(idx)
    print(f'{probs[0, idx.item()]:.5f} : {id2label[str(idx.item())]}', end='\n')

  ### Visualize all the 12 or 24 attention maps separately
  for i, v in enumerate(joint_attentions):
    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask_result = cv2.resize(mask / mask.max(), (original_image.shape[0],original_image.shape[0]))
    mask = cv2.resize(mask / mask.max(), (original_image.shape[0],original_image.shape[0]))[..., np.newaxis]
    result = (mask * original_image.numpy()).astype("uint8")
	# if you need plot uncomment this 
    #fig, (ax21, ax22, ax23) = plt.subplots(ncols=3, figsize=(10, 10))
    #ax21.set_title('Original')
    #ax22.set_title('Attention Map_%d Layer' % (i+1))
    #ax23.set_title('Attention Map_%d Layer' % (i+1))
    #_ = ax21.imshow(original_image)
    #_ = ax22.imshow(mask_result)
    #_ = ax23.imshow(result)

    #plt.imsave(save_folder+str(image_number)+'_processed_attn_layer_'+str(i+1)+'.jpg', mask_result)
    #plt.imsave(save_folder+str(image_number)+'_processed_attn_on_im_layer_'+str(i+1)+'.jpg', np.uint8(result))

