# ------------------------------------------------------------------------------
# Based on 
# Author: Agisilaos Chartsias
# Fetched: 25.02.21
# Paper: Disentangled representation learning in cardiac image analysis
# Version: 22.05.20
# Repository: https://github.com/agis85/anatomy_modality_decomposition
# ------------------------------------------------------------------------------
import torch

# costs kl
def kl(args):
    mean, log_var = args
    kl_loss = -0.5 * torch.sum(1 + log_var - torch.square(mean) - torch.exp(log_var), axis=-1)
    return torch.reshape(kl_loss, (-1, 1))

def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    Instead of sampling from Q(z|X), sample eps = N(0,I): z = z_mean + sqrt(var)*eps
    :param args: args (tensor): mean and log of variance of Q(z|X)
    :return:     z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = z_mean.shape[0]#torch.shape(z_mean)[0]
    dim = z_mean.shape[1]#torch.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = torch.normal(mean = 0.0, std = 0.5, size=(batch, dim)).cuda()#, shape=(batch, dim))
    return z_mean + torch.exp(0.5 * z_log_var).cuda() * epsilon