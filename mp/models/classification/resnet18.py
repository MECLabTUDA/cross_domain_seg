# ------------------------------------------------------------------------------
# Example for a small CNN.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch
import torch.nn.functional as F
from mp.models.model import Model
import torchvision.models as models

class Resnet18(Model):
    r"""An CNN for classification."""
    def __init__(self, input_shape=(3, 32, 32), output_shape=10):
        super().__init__(input_shape, output_shape)
        self.model = models.resnet18(pretrained=False)
        in_ftr_last  = self.model.fc.in_features
        self.model.fc = nn.Linear(in_ftr_last,output_shape,bias=True)
        self.output_shape = output_shape
        self.input_shape = input_shape
        #inputNr = input_shape[0] * input_shape[1] * input_shape[2]
        out_ftr_first = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(input_shape[0], out_ftr_first, 3, 1, 1)
        #self.model.features[0] = nn.Conv2d(input_shape[0], 64, 3, 1, 1)

    def forward(self, x):
        #x = x.squeeze(0)
        #x = torch.cat([x, x, x], 0)
        #x = x.unsqueeze(0)
        x = self.model(x)
        return x