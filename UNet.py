import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.mpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.mpool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.mpool3 = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(256, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)
        self.mpool4 = nn.MaxPool2d(2)
        self.conv9 = nn.Conv2d(512, 1024, 3)
        self.conv10 = nn.Conv2d(1024, 1024, 3)
        self.transConv1 = nn.ConvTranspose2d(1024, 512, 2, stride=(2, 2))
        self.conv11 = nn.Conv2d(1024, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.transConv2 = nn.ConvTranspose2d(512, 256, 2, stride=(2, 2))
        self.conv13 = nn.Conv2d(512, 256, 3)
        self.conv14 = nn.Conv2d(256, 256, 3)
        self.transConv3 = nn.ConvTranspose2d(256, 128, 2, stride=(2, 2))
        self.conv15 = nn.Conv2d(256, 128, 3)
        self.conv16 = nn.Conv2d(128, 128, 3)
        self.transConv4 = nn.ConvTranspose2d(128, 64, 2, stride=(2, 2))
        self.conv17 = nn.Conv2d(128, 64, 3)
        self.conv18 = nn.Conv2d(64, 64, 3)
        self.conv19 = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.mpool1(x2)
        x4 = F.relu(self.conv3(x3))
        x5 = F.relu(self.conv4(x4))
        x6 = self.mpool2(x5)
        x7 = F.relu(self.conv5(x6))
        x8 = F.relu(self.conv6(x7))
        x9 = self.mpool3(x8)
        x10 = F.relu(self.conv7(x9))
        x11 = F.relu(self.conv8(x10))
        x12 = self.mpool4(x11)
        x13 = F.relu(self.conv9(x12))
        x14 = F.relu(self.conv10(x13))
        x15 = self.transConv1(x14)
        x11_crop = x11[:, :, 4:-4, 4:-4]
        x8_crop = x8[:, :, 16:-16, 16:-16]
        x5_crop = x5[:, :, 40:-40, 40:-40]
        x2_crop = x2[:, :, 88:-88, 88:-88]
        x16 = torch.concat((x11_crop, x15), dim=1)
        x17 = F.relu(self.conv11(x16))
        x18 = F.relu(self.conv12(x17))
        x19 = self.transConv2(x18)
        x20 = torch.concat((x8_crop, x19), dim=1)
        x21 = F.relu(self.conv13(x20))
        x22 = F.relu(self.conv14(x21))
        x23 = self.transConv3(x22)
        x24 = torch.concat((x5_crop, x23), dim=1)
        x25 = F.relu(self.conv15(x24))
        x26 = F.relu(self.conv16(x25))
        x27 = self.transConv4(x26)
        x28 = torch.concat((x2_crop, x27), dim=1)
        x29 = F.relu(self.conv17(x28))
        x30 = F.relu(self.conv18(x29))
        x31 = self.conv19(x30)

        return x31


def segmentation_map(IMAGE_PATH):
    """
    
    Arguments : Path of the image 
    
    Returns : The segmentation map as a Torch Tensor
    
    """
    img = Image.open(IMAGE_PATH)
    #Handle .png files here
    if '.png' in IMAGE_PATH:
        img = Image.new('RGB', img.size, (255, 255, 255))
        img.paste(img, mask=img.split()[3])
    img = np.asarray(img)
    #According to the paper
    new_size = (572, 572)
    img = cv2.resize(img, new_size)
    #Transform image to Tensor
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = transform_to_tensor(img)
    input_tensor = input_tensor.unsqueeze(0)
    #Define model here
    model = UNet()
    segmentation_map = model(input_tensor)
    #Option to convert it to image
    transform_to_image = transforms.ToPILImage()
    segmentation_img = transform_to_image(segmentation_map)
    return (segmentation_map, segmentation_img)
