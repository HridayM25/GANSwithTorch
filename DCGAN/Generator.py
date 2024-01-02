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

Z = torch.randn(c.BATCH_SIZE, c.LATENT_DIM, 1, 1)
Thus the input is a 4D tensor of shape (64, 100, 1, 1)
The first step as mentioned in the paper is to obtain a tensor of shape (64, 1024, 4, 4)
This is done by passing the input through a transposed convolutional layer with 1024 filters of size 4x4 and stride 1

"""

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.tConv1 = nn.ConvTranspose2d(c.LATENT_DIM, 1024, 4, stride=(1, 1))
        self.tConv1_bn = nn.BatchNorm2d(1024)
        self.tConv2 = nn.ConvTranspose2d(1024, 512, 4, stride=(2, 2), padding=(1, 1))
        self.tConv2_bn = nn.BatchNorm2d(512)
        self.tConv3 = nn.ConvTranspose2d(512, 256, 4, stride=(2, 2), padding=(1, 1))
        self.tConv3_bn = nn.BatchNorm2d(256)
        self.tConv4 = nn.ConvTranspose2d(256, 128, 4, stride=(2, 2), padding=(1, 1))
        self.tConv4_bn = nn.BatchNorm2d(128)
        self.tConv5 = nn.ConvTranspose2d(128, 64, 4, stride=(2, 2), padding=(1, 1))
        self.tConv5_bn = nn.BatchNorm2d(64)
        self.tConv6 = nn.ConvTranspose2d(64, 3, 4, stride=(2, 2), padding=(1, 1))
        self.tConv6_bn = nn.BatchNorm2d(3)
        
    def forward(self, x):
        x = self.tConv1(x)
        x = self.tConv1_bn(x)
        x = nn.ReLU()(x)
        x = self.tConv2(x)
        x = self.tConv2_bn(x)
        x = nn.ReLU()(x)
        x = self.tConv3(x)
        x = self.tConv3_bn(x)
        x = nn.ReLU()(x)
        x = self.tConv4(x)
        x = self.tConv4_bn(x)
        x = nn.ReLU()(x)
        x = self.tConv5(x)
        x = self.tConv5_bn(x)
        x = nn.ReLU()(x)
        x = self.tConv6(x)
        x = self.tConv6_bn(x)
        x = nn.Tanh()(x)
        return x