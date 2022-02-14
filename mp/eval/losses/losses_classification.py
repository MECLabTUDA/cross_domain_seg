# ------------------------------------------------------------------------------
# Similitude metrics between output and target.
# ------------------------------------------------------------------------------

import torch.nn as nn
from mp.eval.losses.loss_abstract import LossAbstract
import numpy as np
import torch

class CrossEntropyLoss(LossAbstract):
    r"""L1 distance loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, output, target):
        return self.ce(output, target)

class L1Loss(LossAbstract):
    r"""L1 distance loss."""
    def __init__(self, device='cuda:0'):
        super().__init__(device=device)
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, output, target):
        l1loss = self.l1(output, target)
        return l1loss