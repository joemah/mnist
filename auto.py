from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import MNIST
from copy  import deepcopy
from tqdm import tqdm 
from torch import nn
import numpy as np 
import random 
import torch
import os 

class AutoEncoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        

        # encoder
        self.encodcnn = nn.Sequential(
            nn.Conv2d(1,8,3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Conv2d(8,16,3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Conv2d(16,32,3,stride=2,padding=0),
            nn.ReLU(True)
        )
        self.encodlin = nn.Sequential(
            nn.Linear(3*3*32,64),
            nn.ReLU(True),
            nn.Linear(64,encoded_space_dim)
        )

        # decoder
        self.decodelin = nn.Sequential(
            nn.Linear(encoded_space_dim,64),
            nn.ReLU(True),
            nn.Linear(64,3*3*32),
            nn.ReLU(True)
        )
        self.decodcnn  = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,stride=2,output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8,1,3,stride=2,padding=1,output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
        
    def encode(self, x):
        x = self.encodcnn(x)
        x = x.view([x.size(0),-1]) # flatten
        x = self.encodlin(x)
        return x
    
    def decode(self, x):
        x = self.decodelin(x) # apply lin layers
        x = x.view([-1, 32, 3, 3]) # reshape
        x = self.decodcnn(x)
        x = torch.sigmoid(x)
        return x




