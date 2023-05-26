#Importing required Packages
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np

import helper
import zipfile
from PIL import Image

import time
import seaborn as sns
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


model = models.densenet201(weights='DenseNet201_Weights.DEFAULT')

#Freeze denseblock layers for retraining, Optional
for name, child in model.features.named_children():
    if name in ['conv0', 'norm0','relu0','pool0','denseblock1','transition1','denseblock2','transition2','transition3','norm5']:
        #print(name + ' is frozen')
        for param in child.parameters():
            param.requires_grad = False

    else:
        #print(name + ' is unfrozen')
        for param in child.parameters():
            param.requires_grad = True

            
# * Define a new, untrained feed-forward network as a classifier  
class ClassifierH2(nn.Module):
    def __init__(self, inp = 1920, h1=1024, output = 102, drop=0.35):
        super().__init__()
        self.adaptivePool = nn.AdaptiveAvgPool2d((1,1))
        self.maxPool = nn.AdaptiveMaxPool2d((1,1))
        
        self.fla = nn.Flatten()
        self.batchN0 = nn.BatchNorm1d(inp*2,eps=1e-05, momentum=0.1, affine=True)
        self.dropout0 = nn.Dropout(drop)
        self.fc1 = nn.Linear(inp*2, h1)
        self.batchN1 = nn.BatchNorm1d(h1,eps=1e-05, momentum=0.1, affine=True)
        self.dropout1 = nn.Dropout(drop)

        self.fc3 = nn.Linear(h1, output)
        
    def forward(self, x):
        adaptivePool = self.adaptivePool(x)
        maxPool = self.maxPool(x)
        x = torch.cat((adaptivePool,maxPool),dim=1)
        x = self.fla(x)
        x = self.batchN0(x)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.batchN1(x)
        x = self.dropout1(x)         
        x = self.fc3(x)
        
        return x
