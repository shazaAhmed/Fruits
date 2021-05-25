import pickle
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
#matplotlib inline
from tqdm.notebook import tqdm
import torchvision.models as models
from sklearn.model_selection import train_test_split



class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                      
        loss = F.cross_entropy(out, labels)                   # Calculate training loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                                    # Generate predictions
        loss = F.cross_entropy(out, labels)                   # Calculate validation loss
        acc = accuracy(out, labels)                           # Calculate accuracy
        return {'val_loss': loss.detach(),  'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()         # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()            # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.10f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


class SimpleCNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        
        self.res = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                    kernel_size=3, stride=1, padding=1),  # 3 x 100 x 100 -> 32 x 100 x 100
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),  # bs x 32 x 100 x 100 -> 64 x 100 x 100
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),  # bs x 64 x 100 x 100 -> 128 x 100 x 100
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # bs x 128 x 100 x 100 -> 256 x 50 x 50
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),  # bs x 256 x 50 x 50 -> 512 x 25 x 25
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 2, 1),  # bs x 512 x 25 x 25 -> 1024 x 12 x 12
            nn.ReLU(),
            nn.MaxPool2d(4),                          # -> 1024 x 3 x 3
            nn.Flatten(),                             # -> 9216
            nn.Linear(9216, 131)
        )
    
    def forward(self, xb):
        out = self.res(xb)
        return out


class ResNetCNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)     # You can change the resnet model here
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 131)          # Output classes
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True



def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels),     # Batch Normalization
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class CustomCNN(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 128)                                 # 3 x 100 x 100 ->
        self.conv2 = conv_block(128, 256, pool=True)                              # 128 x 100 x 100 ->
        self.res1 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))     # 256 x 50 x 50 ->
        
        self.conv3 = conv_block(256, 512, pool=True)                              # -> 512 x 25 x 25
        self.conv4 = conv_block(512, 1024, pool=True)                             # -> 1024 x 12 x 12
        self.res2 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024)) # -> 1024 x 12 x 12
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),                          # -> 1024 x 3 x 3
                                        nn.Flatten(),                             # -> 9216
                                        nn.Linear(9216, num_classes))             # -> 131
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out    # Residual Block 
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out    # Residual Block
        out = self.classifier(out)
        return out




def predict_image(img,model):
    # Convert to a batch of 1
    xb = img.unsqueeze(0)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    #return valid_ds.classes[preds[0].item()]
    return preds[0].item()

df_2 = pd.read_csv('NutritionalFacts_Fruit_Vegetables_Seafood.csv', engine='python')
def get_nutritional_value_2(fruit_name):
    list_name = fruit_name.split(' ')
    search_term = fruit_name
    if len(list_name) > 1:
        search_term = list_name[0]
    
    return (df_2[df_2['Food and Serving'].str.contains(search_term, na=False)])

