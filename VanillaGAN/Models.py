import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

#Keeping a consistent output dimension
OUTPUT_DIMENSION = 128

class Generator(nn.Module):
    def __init__(self,NOISE_DIMENSION, OUTPUT_DIMENSION):
        super(Generator, self).__init__()
        self.NOISE_DIMENSION = NOISE_DIMENSION
        self.OUTPUT_DIMENSION = OUTPUT_DIMENSION
        self.layer1 = nn.Linear(self.NOISE_DIMENSION, 128)
        self.layer2 = nn.Linear(128,256)
        self.layer3 = nn.Linear(256, self.OUTPUT_DIMENSION)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x
    
class Discriminator(nn.Module):
    def __init__(self, INPUT_DIMENSION):
        super(Discriminator,self).__init__()
        self.INPUT_DIMENSION = INPUT_DIMENSION
        self.layer1 = nn.Linear(self.INPUT_DIMENSION, 1)
    
    def forward(self,x):
        x = F.sigmoid(self.layer1(x))
        return x
        
        
        
    
