import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *


class Net(nn.Module):
    def __init__(self, num_channels, feat):
        super(Net, self).__init__()

        self.conv0_1 = DoubleConv(num_channels, 32)
        self.conv1_2 = DownConv(32)
        self.conv2_3 = DownConv(64)
        self.conv3_4 = DownConv(128) 
        self.conv4_5 = DownConv(256) 

        self.conv5_4 = UpCatConv(512) 
        self.conv4_3 = UpCatConv(256) 
        self.conv3_2 = UpCatConv(128)
        self.conv2_1 = UpCatConv(64)

        self.output = ConvBlock(32, feat, 1, 1, 0, activation=None, norm=None)
        #Reconstruction
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def forward(self, x):
        x1 = self.conv0_1(x)
        x2 = self.conv1_2(x1)
        x3 = self.conv2_3(x2)
        x4 = self.conv3_4(x3)
        x5 = self.conv4_5(x4)
        x4 = self.conv5_4(x5, x4)
        x3 = self.conv4_3(x4, x3)
        x2 = self.conv3_2(x3, x2)
        x1 = self.conv2_1(x2, x1)
        
        return self.output(x1)

