import constants as c
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

"""
The input here is a tensor of shape (64, 3, 64, 64)

"""

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, stride=(2, 2), padding=(1, 1))
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=(2, 2), padding=(1, 1))
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=(2, 2), padding=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, stride=(2, 2), padding=(1, 1))
        self.conv4_bn = nn.BatchNorm2d(512)
        #Not very sure about this part as the paper mentioned no FC layers
        self.fc1 = nn.Linear(512*4*4, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = nn.LeakyReLU(0.2)(x)
        x = x.view(-1, 512*4*4)
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        return x