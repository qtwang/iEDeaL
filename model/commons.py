# coding = utf-8

from typing import Union, List

import torch
# import numpy as np
from torch import nn, Tensor, tanh, Size

from util.conf import Conf
from model.activate import LeCunTanh
from model.normalize import AdaNorm


class Mean(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super(Mean, self).__init__()
            

    def forward(self, input: Tensor) -> Tensor:
        return torch.mean(input)


class Sum(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super(Sum, self).__init__()
            

    def forward(self, input: Tensor) -> Tensor:
        return torch.sum(input)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape


    def forward(self, input: Tensor) -> Tensor:
        return input.view(self.shape)


class Squeeze(nn.Module):
    def __init__(self, dim = None):
        super(Squeeze, self).__init__()
        self.dim = dim


    def forward(self, input: Tensor) -> Tensor:
        if self.dim is None:
            return input.squeeze()
        else:
            return input.squeeze(dim=self.dim)


# depth starts from 1
def getDilation(conf: Conf, depth: int) -> int:
    dilation_type = conf.getHP('dilation_type')

    if dilation_type == 'exponential':
        return int(2 ** (depth - 1))
    elif dilation_type == 'linear':
        return conf.getHP('dilation_base') + conf.getHP('dilation_slope') * (depth - 1)
    
    return conf.getHP('dilation_constant')


def getActivation(conf: Conf, key: str) -> nn.Module:
    name = conf.getHP(key)

    if name == 'tanh':
        return nn.Tanh()
    elif name == 'lecuntanh':
        return LeCunTanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(conf.getHP('relu_slope'))

    return nn.Identity()


def getWeightNorm(conf: Conf, model: nn.Module) -> nn.Module:
    weightnorm_type = conf.getHP('weightnorm_type')

    if weightnorm_type == 'weightnorm':
        return nn.utils.weight_norm(model, dim=conf.getHP('weightnorm_dim'))

    return model


def getLayerNorm(conf: Conf, shape: Union[int, List[int], Size]) -> nn.Module:
    layernorm_type = conf.getHP('layernorm_type')

    if layernorm_type == 'layernorm':
        return nn.LayerNorm(shape, elementwise_affine=conf.getHP('layernorm_elementwise_affine'))
    elif layernorm_type == 'adanorm':
        return AdaNorm(shape, conf.getHP('adanorm_k'), conf.getHP('adanorm_scale'), conf.getHP('eps'), conf.getHP('layernorm_elementwise_affine'))

    return nn.Identity()
