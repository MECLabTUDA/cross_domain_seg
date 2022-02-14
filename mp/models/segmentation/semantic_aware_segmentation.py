from mp.models.model import Model
import torch.nn as nn
import torch
import torch.nn.functional as F

class SA_Segmentation_Model(Model):

    def __init__(self, input_shape=(1, 96, 96), num_labels=2, config=None):
        super().__init__()

        layers = [
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=1, dilated=2),
            ResidualBlock(256, 512, stride=1, dilated=4),
        ]

        self.layers = nn.Sequential(*layers)
        self.dilatedConv6 = nn.Sequential(nn.Conv2d(512, num_labels, kernel_size=3, dilation=6, padding=5), nn.BatchNorm2d(num_labels), nn.ReLU(num_labels))
        self.dilatedConv12 = nn.Sequential(nn.Conv2d(512, num_labels, kernel_size=3, dilation=12, padding=11), nn.BatchNorm2d(num_labels), nn.ReLU(num_labels))
        self.dilatedConv18 = nn.Sequential(nn.Conv2d(512, num_labels, kernel_size=3, dilation=18, padding=17), nn.BatchNorm2d(num_labels), nn.ReLU(num_labels))
        self.dilatedConv24 = nn.Sequential(nn.Conv2d(512, num_labels, kernel_size=3, dilation=24, padding=23), nn.BatchNorm2d(num_labels), nn.ReLU(num_labels))

        up_layers = [
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(256), # 16
            nn.ConvTranspose2d(256, num_labels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_labels),
            nn.ReLU(num_labels), #32
        ]

        self.up_layers = nn.Sequential(*up_layers)

        self.upsample = nn.Upsample(size=(512, 512))
    
    def forward(self, x):

        x = self.layers(x)
        x6 = self.dilatedConv6(x)
        x12 = self.dilatedConv12(x)
        x18 = self.dilatedConv18(x)
        x24 = self.dilatedConv24(x)

        x = x6 + x12 + x18 + x24

        x = self.upsample(x)

        return x

# ------------------------------------------------------------------------------
# Based on 
# Author: yunyei
# Fetched: 13.04.21
# Version: 27.01.20
# Repository: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
# ------------------------------------------------------------------------------
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1, dilated=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False, dilation=dilated)

class ResidualBlock(Model):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilated=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, dilated=dilated)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride, dilated=dilated), nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
