# ------------------------------------------------------------------------------
# Based on 
# Author: Luan Tran
# Fetched: 14.12.20
# Paper: Disentangled Representation Learning GAN for Pose-Invariant Face Recognition (DR-GAN)
# Version: 30.12.19
# Repository: https://github.com/tranluan/DR-GAN
# ------------------------------------------------------------------------------

from mp.models.model import Model
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import torch.nn.init as init
import os
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from mp.eval.inference.predict import softmax

def Tensor2Image(img):
    """
    input (FloatTensor)
    output (PIL.Image)
    """
    img = img.cpu()
    img = transforms.ToPILImage()(img)
    return img
    

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Conv2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)

class conv_unit(Model):
    def __init__(self, in_channels, out_channels, pooling=False):
        super(conv_unit, self).__init__()

        if pooling:
            layers = [nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_channels, out_channels, 3, 2, 0)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]
        
        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)
    
    def forward(self, input):
        x = self.layers(input)
        return x

class Fconv_unit(Model):
    def __init__(self, in_channels, out_channels, unsampling=False):
        super(Fconv_unit, self).__init__()

        if unsampling:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1), nn.ZeroPad2d([0, 1, 0, 1])]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Decoder(Model):
    def __init__(self, N_p=2, N_z = 50, smaller = False, bsize = 6):
        super(Decoder, self).__init__()
        if smaller is False:
            Fconv_layers = [
                Fconv_unit(320, 160),                   # 320x6x6
                Fconv_unit(160, 256),                  
                Fconv_unit(256, 256, unsampling=True)]
        else:
            Fconv_layers = []
  
        Fconv_layers.append(Fconv_unit(256, 128))                   
        Fconv_layers.append(Fconv_unit(128, 192))                   
        Fconv_layers.append(Fconv_unit(192, 192, unsampling=True))
        Fconv_layers.append(Fconv_unit(192, 96))                    
        Fconv_layers.append(Fconv_unit(96, 128))                    
        Fconv_layers.append(Fconv_unit(128, 128, unsampling=True))  
        Fconv_layers.append(Fconv_unit(128, 64))                    
        Fconv_layers.append(Fconv_unit(64, 64))                     
        Fconv_layers.append(Fconv_unit(64, 64, unsampling=True))    
        Fconv_layers.append(Fconv_unit(64, 32))                     
        Fconv_layers.append(Fconv_unit(32, 1))                       # 32 -> 3

        self.smaller = smaller
        self.bsize = bsize

        self.Fconv_layers = nn.Sequential(*Fconv_layers)
        if smaller is False:
            self.fc = nn.Linear(320 + N_p + N_z, 320 * self.bsize * self.bsize)
        else:
            self.fc = nn.Linear(256 + N_p + N_z, 256 * self.bsize * self.bsize)

    def forward(self, input):
        x = self.fc(input)
        if self.smaller is False:
            x = x.view(-1, 320, self.bsize, self.bsize)
        else:
            x = x.view(-1, 256, self.bsize, self.bsize)
        x = self.Fconv_layers(x)
        return x

class Encoder(Model):
    def __init__(self, smaller = False):
        super(Encoder, self).__init__()
        conv_layers = [
            conv_unit(1, 32),                  # 3 -> 32 
            conv_unit(32, 64),                  
            conv_unit(64, 64, pooling=True),    
            conv_unit(64, 64),                  
            conv_unit(64, 128),                 
            conv_unit(128, 128, pooling=True),  
            conv_unit(128, 96),                 
            conv_unit(96, 192),                 
            conv_unit(192, 192, pooling=True),  
            conv_unit(192, 128),                
            conv_unit(128, 256), ]
        if smaller is False:
            conv_layers.append(conv_unit(256, 256, pooling=True))
            conv_layers.append(conv_unit(256, 160))                
            conv_layers.append(conv_unit(160, 320))  
            conv_layers.append(nn.AvgPool2d(kernel_size=12))
        else:
            conv_layers.append(nn.AvgPool2d(kernel_size=6))

        self.smaller = smaller

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, input):
        x = self.conv_layers(input)
        if self.smaller is False:
            x = x.view(-1, 320)
        else:
            x = x.view(-1, 256)
        return x

class Generator(Model):
    def __init__(self, N_p = 2, N_z = 50, smaller = False, bsize = 6):
        super(Generator, self).__init__()
        self.enc = Encoder(smaller = smaller)
        self.dec = Decoder(N_p, N_z, smaller = smaller, bsize = bsize)

    def forward(self, input, domain, noise):
        x = self.enc(input)
        x = torch.cat((x, domain, noise), 1)
        x = self.dec(x)
        return x

    def forward_z(self, input, domain, noise):
        enc = self.enc(input)
        x = torch.cat((enc, domain, noise), 1)
        x = self.dec(x)
        return enc, x

class Discriminator(Model):
    def __init__(self, N_p = 2, N_d = 500, smaller = False, firstLayerFeatureDepth=1):
        super(Discriminator, self).__init__()
        conv_layers = [
            conv_unit(firstLayerFeatureDepth, 32),                  # 3 -> 32 
            conv_unit(32, 64),                  
            conv_unit(64, 64, pooling=True),    
            conv_unit(64, 64),                  
            conv_unit(64, 128),                 
            conv_unit(128, 128, pooling=True),  
            conv_unit(128, 96),                 
            conv_unit(96, 192),                 
            conv_unit(192, 192, pooling=True),  
            conv_unit(192, 128),                
            conv_unit(128, 256), ]
        if smaller is False:
            conv_layers.append(conv_unit(256, 256, pooling=True))
            conv_layers.append(conv_unit(256, 160))                
            conv_layers.append(conv_unit(160, 320))
            conv_layers.append(nn.AvgPool2d(kernel_size=12))    
            # TOD: Kernel size = 6
        else: 
            conv_layers.append(nn.AvgPool2d(kernel_size=6))

        self.conv_layers = nn.Sequential(*conv_layers)

        self.smaller = smaller
        if smaller is False:
            self.fc = nn.Linear(320, N_d + N_p + 1)
        else:
            self.fc = nn.Linear(256, N_d + N_p + 1)

    def forward(self, input):
        x = self.conv_layers(input)
        if self.smaller is False:
            x = x.view(-1, 320)
        else:
            x = x.view(-1, 256)
        x = self.fc(x)
        return x

class CDD_GAN(Model):
    def __init__(self, number_domain = 2, number_identity = 180, number_noise = 50, batch_size=8, config=None, segmentor=None):
        input_shape = config['input_shape']
        super().__init__(input_shape, output_shape=input_shape)

        if config is None:
            config = {}
            config['lr_G'] = 0.0001
            config['lr_D'] = 0.0001
            config['beta1'] = 0.5
            config['beta2'] = 0.999
            config['w_L1'] = 1
            config['device'] = "Unknown"

        # Drop one layer if smaller than 96
        self.bsize = int(input_shape[1] / 16)
        if input_shape[1] < 96:
            self.smaller = True
            self.bsize *= 2
        else:
            self.smaller = False

        self.config = config
        self.batch_size = number_domain

        self.segmentor = segmentor

        self.G = Generator(N_p = number_domain, N_z = number_noise, smaller=self.smaller, bsize = self.bsize)
        self.D = Discriminator(N_p = number_domain, N_d = number_identity, smaller=self.smaller)

        if self.config['UseMaskDiscriminator'] == True:
            self.D_Mask = Discriminator(N_p = 0, N_d = 0, smaller=self.smaller, firstLayerFeatureDepth=2)

        self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.config['lr_G'], betas=(self.config['beta1'], self.config['beta2'])) # set options

        if self.config['UseMaskDiscriminator'] == True:
            self.optimizer_D = optim.Adam(list(self.D.parameters()) + list(self.D_Mask.parameters()), lr=self.config['lr_D'], betas=(self.config['beta1'], self.config['beta2']))
        else:
            self.optimizer_D = optim.Adam(self.D.parameters(), lr=self.config['lr_D'], betas=(self.config['beta1'], self.config['beta2']))

        self.criterion = nn.CrossEntropyLoss()
        self.b_criterion = nn.BCEWithLogitsLoss()
        self.L1_criterion = nn.L1Loss()
        self.config['w_L1'] = 1
        self.w_L1 = self.config['w_L1']

        self.N_z = number_noise
        self.N_p = number_domain
        self.N_d = number_identity

    def one_hot(self, label, depth):
        out_tensor = torch.zeros(len(label), depth)
        for i, index in enumerate(label):
            out_tensor[i][index] = 1
        return out_tensor

    def load_input(self, input):
        img, gt, domain, identity, name, affine = input
        self.image = []
        self.domain = []
        self.identity = []
        self.name = []
        self.affine = []
        for i in range(len(domain)):
            self.image.append(img[i])
            self.domain.append(domain[i])
            self.identity.append(identity[i])
            self.name.append(name[i]) #input['name'][i]
            self.affine.append(affine[i])

    def set_input(self, inputs, randomOut = True, outDomain = None):
        self.load_input(inputs)
        self.image = torch.stack(self.image, dim = 0)
        self.batchsize = len(self.domain)
        if randomOut is True:
            self.target_domain = torch.LongTensor(np.random.randint(self.N_p, size = self.batchsize))
        elif outDomain is not None:
            self.target_domain = torch.LongTensor(np.repeat(self.domain, repeats=self.batchsize))
        else:
            print("ERROR in Out Domain")
            self.target_domain = torch.LongTensor(outDomain)
        
        self.domain = torch.LongTensor(self.domain)

        self.input_domain = self.one_hot(self.target_domain, self.N_p)

        self.cycle_domain = self.one_hot(self.domain, self.N_p)
        self.standard_domain = self.one_hot(torch.LongTensor(np.repeat(self.config['Unet_training_domain'], repeats=self.batchsize)), self.N_p)

        self.identity = torch.LongTensor(self.identity)
        self.fake_identity = torch.zeros(self.batchsize).long() # 0 indicates fake
        self.noise = torch.FloatTensor(np.random.normal(loc=0.0, scale=0.3, size=(self.batchsize, self.N_z)))

        #cuda
        if self.config['device'] != 'cpu':
            self.image = self.image.cuda(self.config['device'])
            self.domain = self.domain.cuda(self.config['device'])
            self.target_domain = self.target_domain.cuda(self.config['device'])
            self.input_domain = self.input_domain.cuda(self.config['device'])
            self.identity = self.identity.cuda(self.config['device'])
            self.fake_identity = self.fake_identity.cuda(self.config['device'])
            self.noise = self.noise.cuda(self.config['device'])
            self.cycle_domain = self.cycle_domain.cuda(self.config['device'])
            self.standard_domain = self.standard_domain.cuda(self.config['device'])

        self.image = Variable(self.image)
        self.domain = Variable(self.domain)
        self.target_domain = Variable(self.target_domain)
        self.input_domain = Variable(self.input_domain)
        self.identity = Variable(self.identity)
        self.fake_identity = Variable(self.fake_identity)
        self.noise = Variable(self.noise)
        self.cycle_domain = Variable(self.cycle_domain)
        self.standard_domain = Variable(self.standard_domain)

    def forward(self, inputs, randomOut = True, outDomain = None):
        self.set_input(inputs, randomOut, outDomain)
        self.syn_image = self.G(self.image, self.input_domain, self.noise)
        self.syn = self.D(self.syn_image)
        self.syn_identity = self.syn[:, :self.N_d+1]
        self.syn_domain = self.syn[:, self.N_d+1:]

        if self.config['UseMaskDiscriminator'] == True:
            # Add Mask
            self.toStandard_image = self.G(self.image, self.standard_domain, self.noise)
            self.toStandard_image_mask = softmax(self.segmentor(self.toStandard_image))
            self.disc_mask_pred = self.D_Mask(self.toStandard_image_mask)

        # Use Cycle Loss
        if self.config['UseCycleLoss'] == True:
            self.cycleResult = self.G(self.syn_image, self.cycle_domain, self.noise)

        self.real = self.D(self.image)
        self.real_identity = self.real[:, :self.N_d+1]
        self.real_domain = self.real[:, self.N_d+1:]

        #assert x.shape == initial_shape
        #return x

    def forward_z(self, inputs):
        self.set_input(inputs, randomOut=False)

        enc, self.syn_image = self.G.forward_z(self.image, self.input_domain, self.noise)

        self.syn = self.D(self.syn_image)
        self.syn_identity = self.syn[:, :self.N_d+1]
        self.syn_domain = self.syn[:, self.N_d+1:]

        self.real = self.D(self.image)
        self.real_identity = self.real[:, :self.N_d+1]
        self.real_domain = self.real[:, self.N_d+1:]

        return enc, self.real_domain

    def init_weights(self):
        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)

    def output_tensor(self, value, depth):
        out = torch.LongTensor(np.repeat(value, repeats = depth))#.reshape((depth, 1))
        if self.config['device'] == 'cpu':
            return out
        else:
            return out.cuda(self.config['device'])

    def output_tensor_eq(self, value, depth):
        out = torch.LongTensor(np.repeat(value, repeats = depth))#.reshape((depth, 1))

        if self.config['device'] == 'cpu':
            out = torch.eq(self.domain, out).type(torch.long)
            return out
        else:
            out = torch.eq(self.domain, out.cuda(self.config['device'])).type(torch.long)
            return out#.cuda(self.config['device'])

    def output_tensor_ce(self, value, depth):
        out = torch.LongTensor(np.repeat(value, repeats = depth)).type(torch.float)#.reshape((depth, 1))
        if self.config['device'] == 'cpu':
            return out
        else:
            return out.cuda(self.config['device'])

    def output_tensor_ce_eq(self, value, depth):
        out = torch.LongTensor(np.repeat(value, repeats = depth))#.reshape((depth, 1))

        if self.config['device'] == 'cpu':
            out = torch.eq(self.domain, out).type(torch.float)
            return out
        else:
            out = torch.eq(self.domain, out.cuda(self.config['device'])).type(torch.float)
            return out#.cuda(self.config['device'])

    def backward_G(self):
        self.Loss_G_syn_identity = self.criterion(self.syn_identity, self.identity)
        
        self.Loss_G_syn_domain = self.criterion(self.syn_domain, self.target_domain)
        self.L1_Loss = self.L1_criterion(self.syn_image, self.image)

        if self.config['UseMaskDiscriminator'] == True:
            if self.config['MaskCELoss'] == True:
                self.Loss_G_syn_mask = self.b_criterion(torch.squeeze(self.disc_mask_pred), self.output_tensor_ce(1, self.batchsize))
            else:
                self.Loss_G_syn_mask = self.L1_criterion(self.disc_mask_pred, self.output_tensor(1, self.batchsize)) 

            if self.config['UseCycleLoss'] == True:
                self.Cycle_Loss = self.L1_criterion(self.cycleResult, self.image)
                self.Loss_G = self.Loss_G_syn_identity + self.Loss_G_syn_domain + self.w_L1 * self.L1_Loss + self.config['CycleLossFactor'] * self.Cycle_Loss + self.Loss_G_syn_mask
            else:
                self.Loss_G = self.Loss_G_syn_identity + self.Loss_G_syn_domain + self.w_L1 * self.L1_Loss + self.Loss_G_syn_mask
        else:
            if self.config['UseCycleLoss'] == True:
                self.Cycle_Loss = self.L1_criterion(self.cycleResult, self.image)
                self.Loss_G = self.Loss_G_syn_identity + self.Loss_G_syn_domain + self.w_L1 * self.L1_Loss + int(self.config['CycleLossFactor']) * self.Cycle_Loss
            else:
                self.Loss_G = self.Loss_G_syn_identity + self.Loss_G_syn_domain + self.w_L1 * self.L1_Loss

        self.Loss_G.backward(retain_graph=True)

    def checkIfEqual(self, standardDomain, id):
        out = torch.eq(standardDomain, 2).type(torch.FloatTensor)
        if self.config['device'] == 'cpu':
            return out
        else:
            return out.cuda(self.config['device'])

    def backward_D(self):
        self.Loss_D_real_identity = self.criterion(self.real_identity, self.identity)
        self.Loss_D_real_domain = self.criterion(self.real_domain, self.domain)

        self.Loss_D_syn = self.criterion(self.syn_identity, self.fake_identity)

        # ----- MASK DESC -----
        if self.config['UseMaskDiscriminator'] == True:
            self.toStandard_disc = self.D_Mask(softmax(self.segmentor(self.image)))
            if self.config['MaskCELoss'] == True:
                self.Loss_D_mask_real = self.b_criterion(torch.squeeze(self.toStandard_disc) * self.output_tensor_ce_eq(self.config['Unet_training_domain'], self.batchsize), self.output_tensor_ce_eq(self.config['Unet_training_domain'], self.batchsize))
                self.Loss_D_mask_fake = self.b_criterion(torch.squeeze(self.disc_mask_pred), self.output_tensor_ce(0, self.batchsize))                 
            else:
                self.Loss_D_mask_real = self.L1_criterion(self.toStandard_disc * self.output_tensor_eq(self.config['Unet_training_domain'], self.batchsize), self.output_tensor_eq(self.config['Unet_training_domain'], self.batchsize)) 
                self.Loss_D_mask_fake = self.L1_criterion(self.disc_mask_pred, self.output_tensor(0, self.batchsize)) 
        # ---------------------

        if self.config['UseMaskDiscriminator'] == True:
            self.Loss_D = self.Loss_D_real_identity + self.Loss_D_real_domain + self.Loss_D_syn + self.Loss_D_mask_fake + self.Loss_D_mask_real
        else:
            self.Loss_D = self.Loss_D_real_identity + self.Loss_D_real_domain + self.Loss_D_syn     
                   
        self.Loss_D.backward()

    def optimize_DG_parameters(self, update_Weights=True):
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        self.backward_G()
        self.backward_D()
        if update_Weights is True:
            self.optimizer_G.step()
            self.optimizer_D.step()

        return self.Loss_G.item(), self.Loss_D.item()

    def optimize_G_parameters(self, update_Weights=True):
        self.optimizer_G.zero_grad()
        self.backward_G()
        if update_Weights is True:
            self.optimizer_G.step()

        return self.Loss_G.item(), self.Loss_D.item()

    def optimize_D_parameters(self, update_Weights=True):
        self.optimizer_D.zero_grad()
        self.backward_D()
        if update_Weights is True:
            self.optimizer_D.step()

    def save_result(self, result_dir, epoch=None, prefix=None):
        for i, syn_img in enumerate(self.syn_image.data):
            if i < 3:
                # Save Img
                img = self.image.data[i]
                filename = self.name[i] 

                if epoch:
                    filename = 'epoch{0}_{1}_{2}'.format(epoch, prefix, filename) + '.png'

                path = os.path.join(result_dir, filename)
                img = Tensor2Image(img)
                syn_img = Tensor2Image(syn_img)

                width, height = img.size
                result_img = Image.new(img.mode, (width*2, height))
                result_img.paste(img, (0, 0, width, height))
                result_img.paste(syn_img, box=(width, 0))
                result_img.save(path)
        if self.config['UseMaskDiscriminator'] == True:
            for i, mask_img in enumerate(self.toStandard_image_mask.data):
                if i < 3:
                    # Save mask
                    img = self.toStandard_image.data[i]
                    filename = self.name[i] + "_mask" 

                    if epoch:
                        filename = 'epoch{0}_{1}_{2}'.format(epoch, prefix, filename) + '.png'

                    path = os.path.join(result_dir, filename)
                    img = Tensor2Image(img)
                    mask_img = Tensor2Image(mask_img)

                    width, height = img.size
                    result_img = Image.new(img.mode, (width*2, height))
                    result_img.paste(img, (0, 0, width, height))
                    result_img.paste(mask_img, box=(width, 0))
                    result_img.save(path)

