# coding = utf-8

import imp
from torch import nn

from util.conf import Conf
from model.resnet import ResNet, ResFR, Res1d18
from model.inception import InceptionTime


def getModel(conf: Conf) -> nn.Module:
    model_name = conf.getHP('model')
    
    if model_name == 'resnet':
        return ResNet(conf)
    elif model_name == 'resnetfr':
        return ResFR(conf)
    elif model_name == 'res1d18':
        return Res1d18(conf)
    elif model_name == 'incept':
        return InceptionTime(conf)

    raise ValueError('invalid model name: {:s}'.format(model_name))
