# ------------------------------------------------------------------------------
# Based on 
# Author: Agisilaos Chartsias
# Fetched: 25.02.21
# Paper: Disentangled representation learning in cardiac image analysis
# Version: 22.05.20
# Repository: https://github.com/agis85/anatomy_modality_decomposition
# ------------------------------------------------------------------------------
from mp.models.model import Model
from mp.models.segmentation.unet_fepegar import UNet2D
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from mp.models.sdnet.sdnet_utils import kl, sampling
import numpy as np

class SD_Net(Model):
    def __init__(self, input_shape=(1, 64, 64), num_labels=2, num_z=8, config=None):
        super().__init__()
        self.config = config

        self.enc_anatomy = Encoder_Anatomy(input_shape=input_shape) 
        self.enc_modality = Encoder_Modality(num_z=num_z, input_shape=input_shape) 
        self.decoder = Decoder(num_labels=num_labels, num_z=num_z, input_shape=input_shape)
        self.segmentor = Segmentor(num_labels)

        self.enc_anatomy = self.enc_anatomy.cuda()
        self.enc_modality = self.enc_modality.cuda()
        self.decoder = self.decoder.cuda()
        self.segmentor = self.segmentor.cuda()

        self.sdnet_optimizer = optim.Adam(
            list(self.enc_anatomy.parameters()) + 
            list(self.enc_modality.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.segmentor.parameters()), 
            lr=0.0001, betas=(0.5, 0.999))

        self.sdnet_optimizer_z = optim.Adam(
            list(self.enc_anatomy.parameters()) + 
            list(self.enc_modality.parameters()) + 
            list(self.decoder.parameters()), 
            lr=0.0001, betas=(0.5, 0.999))


    def forward(self, image):
        fake_s = self.enc_anatomy.forward(image)
        self.fake_z, divergence = self.enc_modality.forward(fake_s, image)

        fake_m = self.segmentor.forward(fake_s)
        self.rec_x = self.decoder.forward(self.fake_z, fake_s)

        return fake_m, self.rec_x, self.fake_z, divergence

    def forward_z(self, image):
        fake_s = self.enc_anatomy.forward(image)
        fake_z, _ = self.enc_modality.forward(fake_s, image)

        return fake_s, fake_z

    def forward_changeStyleToZ(self, image):
        self.fake_s = self.enc_anatomy.forward(image)
        return self.decoder.forward(self.fake_z, self.fake_s)


# Based on a standard U-Net, Ronneberg et al. 2015
class Encoder_Anatomy(Model):
    def __init__(self, input_shape=(1, 64, 64)):
        super().__init__()
        self.downsample = 4

        self.f = 64 
        self.out_channels = 12
        self.input_shape = input_shape

        # define model

        self.unet_downsample_build()
        self.unet_bottleneck()
        self.unet_upsample_build()
        self.out_build()

    def forward(self, inp):
        l = self.unet_downsample(inp)
        l = self.bottleneck(l)
        l = self.unet_upsample(l)
        l = self.out_forward(l)

        return l 

    def unet_downsample_build(self):
        self.udc_l0 = self.unet_conv_block(self.input_shape[0], self.f)
        self.udp_l0 = nn.MaxPool2d(kernel_size=2)

        if self.downsample > 1:
            self.udc_l1 = self.unet_conv_block(self.f, self.f*2)
            self.udp_l1 = nn.MaxPool2d(kernel_size=2)

        if self.downsample > 2:
            self.udc_l2 = self.unet_conv_block(self.f*2, self.f*4)
            self.udp_l2 = nn.MaxPool2d(kernel_size=2)

        if self.downsample > 3:
            self.udc_l3 = self.unet_conv_block(self.f*4, self.f*8)
            self.udp_l3 = nn.MaxPool2d(kernel_size=2)

    def unet_downsample(self, inp):
        self.d_l0 = self.udc_l0(inp)
        l = self.udp_l0(self.d_l0)

        if self.downsample > 1:
            self.d_l1 = self.udc_l1(l)
            l = self.udp_l1(self.d_l1)

        if self.downsample > 2:
            self.d_l2 = self.udc_l2(l)
            l = self.udp_l2(self.d_l2)

        if self.downsample > 3:
            self.d_l3 = self.udc_l3(l)
            l = self.udp_l3(self.d_l3)

        return l 


    def unet_upsample_build(self):
        if self.downsample > 3:
            self.uuu_l3 = self.upsample_block(self.f*8)
            self.uuc_l3 = self.unet_conv_block(self.f*8*2, self.f*8)

        if self.downsample > 2:
            self.uuu_l2 = self.upsample_block(self.f*4)
            self.uuc_l2 = self.unet_conv_block(self.f*4*2, self.f*4)

        if self.downsample > 1:
            self.uuu_l1 = self.upsample_block(self.f*2)
            self.uuc_l1 = self.unet_conv_block(self.f*2*2, self.f*2)

        self.uuu_l0 = self.upsample_block(self.f*1)
        self.uuc_l0 = self.unet_conv_block(self.f*2, self.f)

    def unet_upsample(self, l):
        if self.downsample > 3:
            l = self.uuu_l3(l)
            l = torch.cat([l, self.d_l3], 1)
            l = self.uuc_l3(l)

        if self.downsample > 2:
            l = self.uuu_l2(l)
            l = torch.cat([l, self.d_l2], 1)
            l = self.uuc_l2(l)

        if self.downsample > 1:
            l = self.uuu_l1(l)
            l = torch.cat([l, self.d_l1], 1)
            l = self.uuc_l1(l)

        l = self.uuu_l0(l)
        l = torch.cat([l, self.d_l0], 1)
        l = self.uuc_l0(l)

        return l 

    def unet_bottleneck(self):
        flt = self.f*2
        if self.downsample > 1:
            flt *=2
        if self.downsample > 2:
            flt *=2
        if self.downsample > 3:
            flt *=2
        self.bottleneck = self.unet_conv_block(int(flt/2), flt)

    def unet_conv_block(self, cf, f):
        layers = [
            nn.Conv2d(cf, f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f),
            nn.LeakyReLU(),
            nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f),
            nn.LeakyReLU()
        ]

        return nn.Sequential(*layers) 

    def upsample_block(self, f):
        layers = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(f*2, f, kernel_size=3, padding=1),
            nn.BatchNorm2d(f),
            nn.LeakyReLU()
        ]

        return nn.Sequential(*layers) 

    def out_build(self):
        self.out = nn.Conv2d(self.f, 1, kernel_size=3, padding=1)

    def out_forward(self, l):
        l = self.out(l)
        return l

class Encoder_Modality(Model):
    def __init__(self, num_z, input_shape):
        super().__init__()

        lin_size = self.lin_size(input_shape)

        layers = [
            nn.Conv2d(self.input_shape[0]*2, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(lin_size, 32), 
            nn.BatchNorm1d(32),    
            nn.LeakyReLU()    
        ]

        self.input_shape = input_shape
        self.layers = nn.Sequential(*layers)
        self.num_z = num_z

        self.z_mean_layer = nn.Linear(32, self.num_z)
        self.z_log_var_layer = nn.Linear(32, self.num_z)
        
        self.lambda_layer = LambdaLayer(sampling)
        self.lambda_layer2 = LambdaLayer(kl)
        

    def forward(self, anatomy, image):
        x = torch.cat([anatomy, image], dim=1)
        x = self.layers(x)

        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)

        z = self.lambda_layer([z_mean,z_log_var])
        divergence = self.lambda_layer2([z_mean, z_log_var])
        return [z, divergence]

    # Calculate the amount of vars after flatten
    def lin_size(self, input_shape):
        lin_size = input_shape[2]

        for _ in range(4):
            if lin_size % 2 == 1:
                lin_size -=1
            else: 
                lin_size -=2
            lin_size /= 2
        lin_size = 16 * lin_size * lin_size 
        return int(lin_size)

class Decoder(Model):
    def __init__(self, num_labels, num_z, input_shape):
        super().__init__()
        self.num_labels = num_labels
        self.input_shape = input_shape
        self.num_z = num_z

        spatial_shape = tuple(self.input_shape[:-1]) + (self.num_labels,)
        spatial_input = spatial_shape
        resd_input = (num_z,)

        self.l1 = FiLM_layer(num_labels, num_z, spatial_input, resd_input, first=True)
        self.l2 = FiLM_layer(num_labels, num_z, spatial_input, resd_input)
        self.l3 = FiLM_layer(num_labels, num_z, spatial_input, resd_input)
        self.l4 = FiLM_layer(num_labels, num_z, spatial_input, resd_input)

        self.l5 = nn.Conv2d(self.num_labels, 1, kernel_size=3, padding=1)
    def forward(self, z, s):
        
        l = self.l1(z, s) 
        l = self.l2(z, l)
        l = self.l3(z, l)
        l = self.l4(z, l)

        l = self.l5(l)

        return l 

class Segmentor(Model):
    def __init__(self, num_labels):
        super().__init__()
        conv_channels = num_labels + 1 

        layers = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(),
            nn.Conv2d(64, conv_channels, kernel_size=3, stride=1, padding=1),
            nn.Softmax(), #TODO MAYBE
            LambdaLayer(lambda x: x[:, 0:conv_channels - 1, :, :])
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x
        
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class Conv_Block():
    def __init__(self, in_channels, out_channels, pooling=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channels), nn.LeakyReLU()]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, input):
        x = self.layers(input)
        return x

class Padding_Same(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.pad(x, (0, 0, 2, 1))
        return x

class Dense_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = [nn.Linear(in_channels, out_channels), nn.BatchNorm2d(out_channels), nn.LeakyReLU()]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class FiLM_pred(nn.Module):
    def __init__(self, num_chn, num_z):
        super().__init__()
        self.num_chn = num_chn*2
        self.lin_l1 = nn.Linear(num_z, num_chn) 
        self.rel_l1 = nn.LeakyReLU()
        self.lin_l2 = nn.Linear(num_chn, num_chn)

        self.gamma_l3 = LambdaLayer(lambda x: x[:, :int(num_chn/2)])
        self.beta_l3 = LambdaLayer(lambda x: x[:, int(num_chn/2):])

    def forward(self, z):
        l = self.lin_l1(z)
        l = self.rel_l1(l)
        l = self.lin_l2(l)
        gamma = self.gamma_l3(l)
        beta = self.beta_l3(l)

        return gamma, beta

class FiLM_layer(nn.Module):
    def __init__(self, num_labels, num_z, spatial_input, resd_input, first=False):
        super().__init__()
        self.num_labels = num_labels

        in_size = self.num_labels
        if first:
            in_size = 1
        self.c_l1 = nn.Conv2d(in_size, self.num_labels, kernel_size=3, stride=1, padding=1)
        self.r_l1 = nn.LeakyReLU()

        self.c_l2 = nn.Conv2d(self.num_labels, self.num_labels, kernel_size=3, stride=1, padding=1)

        self.fp_l2 = FiLM_pred(self.num_labels, num_z) 

        num_chn = 2 * self.num_labels

    def forward(self, z, s):
        l1 = self.c_l1(s)
        l1 = self.r_l1(l1)

        l2 = self.c_l2(l1)
        gamma_l2, beta_l2 = self.fp_l2(z)
        l2 = self.FiLM(l2, gamma_l2, beta_l2)
        l2 = nn.LeakyReLU()(l2)
        l = torch.add(l1, l2)
        return l

    def FiLM(self, x, gamma, beta):
        
        gamma = torch.reshape(gamma, (gamma.shape[0], 1, 1, gamma.shape[-1]))
        gamma = torch.Tensor.repeat(gamma, (1, x.shape[1], x.shape[2], 1)) 
                       
        beta = torch.reshape(beta, (beta.shape[0], 1, 1, beta.shape[-1]))
        beta = torch.Tensor.repeat(beta, (1, x.shape[1], x.shape[2], 1))

        return x * gamma + beta
