# Implementation of:
# Semantic-Aware Generative Adversarial Nets for Unsupervised Domain Adaptation in Chest X-ray Segmentation
# https://arxiv.org/pdf/1806.00600.pdf

from mp.models.model import Model
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from mp.models.segmentation.semantic_aware_segmentation import ResidualBlock
import torch.optim as optim
import numpy as np
from torchvision import transforms
import random
import os
from PIL import Image
from mp.eval.inference.predict import softmax

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.uniform_(m.weight.data, 0.02, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)

class SA_Full(Model):
    def __init__(self, input_shape, number_labels, config=None, segmentor = None):
        super(SA_Full, self).__init__()
        
        self.generator_TS = SA_Generator(big=False)
        self.generator_ST = SA_Generator(big=False)

        self.discriminator_S = SA_Discriminator(input_shape)
        self.discriminator_T = SA_Discriminator(input_shape)
        mask_shape = [number_labels, input_shape[1], input_shape[2]]
        self.discriminator_M = SA_Discriminator(mask_shape)

        # initialize weights
        weights_init_normal(self.generator_TS)
        weights_init_normal(self.generator_ST)
        weights_init_normal(self.discriminator_S)
        weights_init_normal(self.discriminator_T)
        weights_init_normal(self.discriminator_M)

        self.segmentor = segmentor

        if config is None:
            config = {}
            config['lr'] = 0.0001
            config['beta1'] = 0.5
            config['beta2'] = 0.999
            config['w_L1'] = 1
            config['device'] = "Unknown"
        
        self.config = config

        self.optimizer_G = optim.Adam(list(self.generator_TS.parameters()) + list(self.generator_ST.parameters()), lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2'])) # set options
        self.optimizer_D = optim.Adam(list(self.discriminator_S.parameters()) + list(self.discriminator_T.parameters()) + list(self.discriminator_M.parameters()), lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        
        self.criterion = nn.MSELoss() #nn.L1Loss()#
        self.L1_criterion = nn.L1Loss()

    def setSegmentor(self, segmentor):
        self.segmentor = segmentor

    def forward(self, input_s, input_t):
        self.depth_s = input_s.size()[0]
        self.depth_t = input_t.size()[0]
        self.s = input_s
        self.t = input_t
        
        self.st = self.generator_ST(input_s)

        self.ts = self.generator_TS(input_t)

        self.sts = self.generator_TS(self.st)

        self.tst = self.generator_ST(self.ts)
        
        self.pred_lab = self.segmentor(self.ts)
        self.pred_lab = softmax(self.pred_lab)

        self.disc_s = self.discriminator_S(self.ts)
        self.disc_t = self.discriminator_T(self.st)

        self.disc_m = self.discriminator_M(self.pred_lab)

    def output_tensor(self, value, depth):
        out = torch.FloatTensor(np.repeat(value, repeats = depth))
        new_shape = (len(out), 1)
        out = out.view(new_shape)

        return out.cuda(self.config['device'])

    def one_hot(self, label, depth):
        out_tensor = torch.zeros(len(label), depth)
        for i, index in enumerate(label):
            out_tensor[i][index] = 1
        return out_tensor

    def optimize_DG_parameters(self, update_Weights=True):

        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        self.backward_G()
        self.backward_D()
        if update_Weights is True:
            self.optimizer_G.step()
            self.optimizer_D.step()

        return self.loss_G.item()

    def backward_DG(self):
        # Generators

        # generators
        loss_st = self.criterion(self.disc_t, self.output_tensor(1, self.depth_s))
        loss_ts = self.criterion(self.disc_s, self.output_tensor(1, self.depth_t))
        # cycle
        loss_sts = self.L1_criterion(self.sts, self.s)
        loss_tst = self.L1_criterion(self.tst, self.t)
        # mask
        loss_m = self.criterion(self.disc_m, self.output_tensor(1, self.depth_t))

        # Discriminators
        real_m = self.segmentor(self.s)
        real_m = softmax(real_m)
        real_m = self.discriminator_M(real_m)
        real_s = self.discriminator_S(self.s)
        real_t = self.discriminator_T(self.t)

        loss_st_real = self.criterion(real_t, self.output_tensor(1, self.depth_s))
        loss_ts_real = self.criterion(real_s, self.output_tensor(1, self.depth_t))
        loss_m_real = self.criterion(real_m, self.output_tensor(1, self.depth_s))

        self.loss_G = (loss_st + loss_st_real) + 0.5 * (loss_ts + loss_ts_real) + 10 * (loss_sts + loss_tst) + 0.5 * (loss_m + loss_m_real)

        self.loss_G.backward(retain_graph=True)

    def backward_G(self):

        # generators
        loss_st = self.criterion(self.disc_t, self.output_tensor(1, self.depth_s))

        loss_ts = self.criterion(self.disc_s, self.output_tensor(1, self.depth_t))
        # cycle
        loss_sts = self.L1_criterion(self.sts, self.s)
        loss_tst = self.L1_criterion(self.tst, self.t)
        # mask
        loss_m = self.criterion(self.disc_m, self.output_tensor(1, self.depth_t))

        # Discriminators
        real_m = self.segmentor(self.s)
        real_m = softmax(real_m)
        real_m = self.discriminator_M(real_m)
        real_s = self.discriminator_S(self.s)
        real_t = self.discriminator_T(self.t)

        self.loss_G = (loss_st) + 0.5 * (loss_ts) + 10 * (loss_sts + loss_tst) + 0.5 * (loss_m)

        self.loss_G.backward(retain_graph=True)

    def optimize_G_parameters(self, update_Weights=True):
        self.optimizer_G.zero_grad()
        self.backward_G()
        if update_Weights is True:
            self.optimizer_G.step()

        return self.loss_g.item()

    def backward_D(self):
        
        # generators
        loss_st = self.criterion(self.disc_t, self.output_tensor(0, self.depth_s))
        loss_ts = self.criterion(self.disc_s, self.output_tensor(0, self.depth_t))
        # mask
        loss_m = self.criterion(self.disc_m, self.output_tensor(0, self.depth_t))

        # Discriminators
        real_m = self.segmentor(self.s)
        real_m = softmax(real_m)
        real_m = self.discriminator_M(real_m)
        real_s = self.discriminator_S(self.s)
        real_t = self.discriminator_T(self.t)


        loss_st_real = self.criterion(real_t, self.output_tensor(1, self.depth_t))
        loss_ts_real = self.criterion(real_s, self.output_tensor(1, self.depth_s))
        loss_m_real = self.criterion(real_m, self.output_tensor(1, self.depth_s))

        self.loss_D = (loss_st + loss_st_real) + 0.5 * (loss_ts + loss_ts_real) + 0.5 * (loss_m + loss_m_real)

        self.loss_D.backward(retain_graph=False)

    def optimize_D_parameters(self, update_Weights=True):
        self.optimizer_D.zero_grad()
        self.backward_D()

        if update_Weights is True:
            self.optimizer_D.step()
        
        return self.loss_d.item()

    def save_result(self, result_dir, epoch=None, prefix=None):
        for x in range(4):

            if x == 0:
                source = self.s
                dest = self.st
                tag = "_st"
            if x == 1:
                source = self.t
                dest = self.ts
                tag = "_ts"
            if x == 2:
                source = self.s
                dest = self.sts
                tag = "_sts"

            if x == 3:
                source = self.s
                dest = self.pred_lab
                tag = "_mask"
                

            for i, syn_img in enumerate(dest.data):
                if i == 0:
                    img = source.data[i]
                    filename = str(random.randint(0, 10000))

                    if epoch:
                        filename = 'epoch{0}_{1}_{2}'.format(epoch, prefix, filename) + tag + '.png'

                    path = os.path.join(result_dir, filename)
                    img = self.Tensor2Image(img)
                    syn_img = self.Tensor2Image(syn_img)

                    width, height = img.size
                    result_img = Image.new(img.mode, (width*2, height))
                    result_img.paste(img, (0, 0, width, height))
                    result_img.paste(syn_img, box=(width, 0))
                    result_img.save(path)

    def Tensor2Image(self, img):
        """
        input (FloatTensor)
        output (PIL.Image)
        """
        img = img.cpu()
        img = transforms.ToPILImage()(img)
        return img




class SA_Generator(Model):
    def __init__(self, big=False):
        super(SA_Generator, self).__init__()
        
        self.encoder = SA_Encoder(big)
        self.transformer = SA_Transformer(big)
        self.decoder = SA_Decoder(big)

    def forward(self, input):
        x = self.encoder(input)
        x = self.transformer(x)
        x = self.decoder(x)

        return x

class SA_Encoder(Model):
    def __init__(self, big):
        super(SA_Encoder, self).__init__()

        if big == False:
            layers = [
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(128)            
            ]
        else:
            layers = [
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(128),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(256),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(512)            
            ]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

class SA_Decoder(Model):
    def __init__(self, big):
        super(SA_Decoder, self).__init__()
        
        if big == False:
            layers = [
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(64),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(32),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(1),
                nn.ReLU(1),
            ]
        else:
            layers = [
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(256),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(128),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(64),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(32),
                nn.ConvTranspose2d(32, 1, 4, 2, 1),
                nn.BatchNorm2d(1),
                nn.ReLU(1),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

class SA_Transformer(Model):
    def __init__(self, big):
        super(SA_Transformer, self).__init__()
        
        if big:
            size = 512
        else:
            size = 128

        layers = [
            ResidualBlock(size, size),
            ResidualBlock(size, size),
            ResidualBlock(size, size),
            ResidualBlock(size, size),
            ResidualBlock(size, size),
            ResidualBlock(size, size),
            ResidualBlock(size, size),
            ResidualBlock(size, size),
            ResidualBlock(size, size),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

class SA_Discriminator(Model):
    def __init__(self, input_shape):
        super(SA_Discriminator, self).__init__()
        
        lastLayers = int(input_shape[1]/32)

        layers = [
            nn.ZeroPad2d([0, 1, 0, 1]),
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(32),
            nn.ZeroPad2d([0, 1, 0, 1]),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(64),
            nn.ZeroPad2d([0, 1, 0, 1]),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(128),
            nn.ZeroPad2d([0, 1, 0, 1]),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(256),
            nn.ZeroPad2d([0, 1, 0, 1]),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(512),
            nn.Flatten(),
            nn.Linear(512 * lastLayers * lastLayers, 1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)