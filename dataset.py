from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import torch 
from torchvision import datasets

os.makedirs("logs/", exist_ok=True)

import logging
logging.basicConfig(filename='logs/network.log', format='%(asctime)s: %(filename)s: %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)



def dataset_stats(sample_data):
  shape = sample_data.shape
  mean = np.mean(sample_data,axis=(0,1,2))/255.
  std = np.std(sample_data,axis=(0,1,2))/255.
  var = np.var(sample_data,axis=(0,1,2))/255.
  return mean,std,var

def visualize_images(sample_data,size):
  row,col  = size[0],size[1]
  figure = plt.figure(figsize=(12,10))
  for i in range(1,col*row+1):
    img,lab = sample_data[i]
    figure.add_subplot(row,col,i)
    plt.title(sample_data.classes[lab])
    plt.axis("off")
    plt.imshow(img)
  plt.tight_layout()
  plt.show()

def visualize_augmentated_images(sample_data,aug_details):
  rows = len(aug_details)
  cols = len(sample_data.classes)
  fig, axes = plt.subplots(cols, rows, figsize=( 3*rows, 15), squeeze=False)
  for i, (key, aug) in enumerate(aug_details.items()):
    for j in range(cols):
      ax = axes[j,i]
      if j == 0:
        ax.text(0.5, 0.5, key, horizontalalignment='center', verticalalignment='center', fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
      else:
        image, label = sample_data[j-1]
        if aug is not None:
          transform = A.Compose([aug])
          image = np.array(image)
          image = transform(image=image)['image']
          
        ax.imshow(image)
        ax.set_title(f'{sample_data.classes[label]}')
        ax.axis('off')

  plt.tight_layout()
  plt.show()
     
class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        
    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label
