# coding = utf-8

import torch
import numpy as np
from torch import nn, Tensor

from util.conf import Conf
from util.commons import discretize
from model.commons import Squeeze, getDilation, getActivation, getWeightNorm, getLayerNorm


class _ResBlock(nn.Module):
    def __init__(self, conf: Conf, in_channels, out_channels, dilation):
        super(_ResBlock, self).__init__()

        activation_key = 'activation_conv'

        dim_series = conf.getHP('dim_series')
        kernel_size = conf.getHP('size_kernel')
        padding = int(kernel_size / 2) * dilation
        bias = conf.getHP('layernorm_type') == 'none'

        self.__residual_link = nn.Sequential(getWeightNorm(conf, nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)),
                                             getLayerNorm(conf, dim_series), 
                                             getActivation(conf, activation_key),

                                             getWeightNorm(conf, nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)),
                                             getLayerNorm(conf, dim_series))
        
        if in_channels != out_channels:
            self.__identity_link = getWeightNorm(conf, nn.Conv1d(in_channels, out_channels, 1, bias=bias))
        else:
            self.__identity_link = nn.Identity()

        self.__after_addition = getActivation(conf, activation_key)
        
        
    def forward(self, input: Tensor) -> Tensor:
        residual = self.__residual_link(input)
        identity = self.__identity_link(input)

        return self.__after_addition(identity + residual)


class _PreActivatedResBlock(nn.Module):
    def __init__(self, conf: Conf, in_channels, out_channels, dilation, first = False, last = False):
        super(_PreActivatedResBlock, self).__init__()

        activation_key = 'activation_conv'

        dim_series = conf.getHP('dim_series')
        kernel_size = conf.getHP('size_kernel')
        padding = int(kernel_size / 2) * dilation
        bias = conf.getHP('layernorm_type') == 'none' or not conf.getHP('layernorm_elementwise_affine')

        if first:
            if self.__conf.getHP('normalize') == 'input':
                self.__first_block = nn.Sequential(getLayerNorm(conf, dim_series),
                                                   getWeightNorm(conf, nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)))
            else:
                self.__first_block = getWeightNorm(conf, nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias))

            in_channels = out_channels
        else:
            self.__first_block = nn.Identity()

        self.__residual_link = nn.Sequential(getLayerNorm(conf, dim_series), 
                                             getActivation(conf, activation_key),
                                             getWeightNorm(conf, nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)),
                                      
                                             getLayerNorm(conf, dim_series),
                                             getActivation(conf, activation_key),
                                             getWeightNorm(conf, nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)))
        
        if in_channels != out_channels:
            self.__identity_link = getWeightNorm(conf, nn.Conv1d(in_channels, out_channels, 1, bias=bias))
        else:
            self.__identity_link = nn.Identity()

        if last:
            self.__after_addition = nn.Sequential(getLayerNorm(conf, dim_series), 
                                                  getActivation(conf, activation_key))
        else:
            self.__after_addition = nn.Identity()
        
        
    def forward(self, input: Tensor) -> Tensor:
        input = self.__first_block(input)

        residual = self.__residual_link(input)
        identity = self.__identity_link(input)

        return self.__after_addition(identity + residual)


class _ResCNN(nn.Module):
    def __init__(self, conf: Conf):
        super(_ResCNN, self).__init__()

        num_resblock = conf.getHP('num_resblock')

        if conf.getHP('dilation_type') == 'exponential':
            assert num_resblock > 1 and 2 ** (num_resblock + 1) <= conf.getHP('dim_series') + 1

        in_channels = conf.getHP('num_input_channels')
        inner_channels = conf.getHP('num_latent_channels')
        out_channels = conf.getHP('num_output_channels')

        if conf.getHP('resblock_pre_activation'):
            ResModule = _PreActivatedResBlock
        else:
            ResModule = _ResBlock

        self.__input = ResModule(conf, in_channels, inner_channels, getDilation(conf, 1))
        self.__latent = nn.Sequential(*[ResModule(conf, inner_channels, inner_channels, getDilation(conf, depth)) for depth in range(2, num_resblock)])
        self.__output = ResModule(conf, inner_channels, out_channels, getDilation(conf, num_resblock))

        self.latent = None

        
    def forward(self, input: Tensor,  auxiliary_input: Tensor = None) -> Tensor:
        latent_in = self.__input(input)

        if auxiliary_input is not None:
            latent_in += auxiliary_input

        self.latent = self.__latent(latent_in)

        return self.__output(self.latent)



class _DropoutInfer(nn.Module):
    def __init__(self, p, device):
        super(_DropoutInfer, self).__init__()

        self._p = torch.tensor(p).to(device)

        
    def forward(self, input: Tensor) -> Tensor:
        return self._p * input



class ResNet(nn.Module):
    def __init__(self, conf: Conf):
        super(ResNet, self).__init__()

        # TODO fixed for binary classification
        assert conf.getHP('num_class') == 2
        num_class = 1

        self.__threshold = conf.getHP('threshold')

        self.__cnn = _ResCNN(conf)

        self.latent = None
        self.latent4label = None

        self.__pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )

        if conf.getHP('dropout'):
            dropout_p = conf.getHP('dropout_p')
            self.__dropout = nn.Dropout(p=dropout_p, inplace=False)
            self.__dropout_infer = _DropoutInfer(p=dropout_p, device=conf.getHP('device'))
        else:
            self.__dropout = nn.Identity()
            self.__dropout_infer = nn.Identity()

        self.__infer = nn.Sequential(
            nn.Linear(conf.getHP('num_output_channels'), num_class),

            # whether Softmax is necessary for binary classification
            # nn.Softmax(dim=1)

            Squeeze(),
            nn.Sigmoid()
        )


    def forward(self, input: Tensor, auxiliary_input: Tensor = None, refine: bool = False, detach: bool = True) -> Tensor:
        latent_out = self.__cnn(input, auxiliary_input)

        self.latent = self.__cnn.latent

        self.latent4label = self.__pool(latent_out)

        return self.__infer(self.__dropout(self.latent4label))


    def infer(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            latent = self.__dropout_infer(self.__pool(self.__cnn(input)))
            predictions = self.__infer(latent).detach().cpu().numpy()

        return discretize(predictions, self.__threshold)



class ResFR(nn.Module):
    def __init__(self, conf: Conf):
        super(ResFR, self).__init__()

        self.__threshold = conf.getHP('threshold')

        self.__filter = ResNet(conf)

        self.__refine = ResNet(conf)


    def forward(self, input: Tensor, refine: bool = False, detach: bool = True) -> Tensor:
        output = self.__filter(input)

        if refine:
            auxiliary_input = self.__filter.latent

            if detach:
                auxiliary_input.detach()

            output = self.__refine(input, auxiliary_input)
        
        return output


    def infer(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            filter_predictions = self.__filter(input).detach().cpu().numpy()

            auxiliary_input = self.__filter.latent

            refine_predictions = self.__filter(input, auxiliary_input).detach().cpu().numpy()

        filter_predictions = discretize(filter_predictions, self.__threshold)
        refine_predictions = discretize(refine_predictions, self.__threshold)

        predictions = np.logical_and(filter_predictions, refine_predictions).astype(int)

        return predictions



class _Res1d18Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Res1d18Block, self).__init__()

        if out_channels > in_channels:
            stride = 2
        else:
            stride = 1

        kernel_size = 3
        padding = int(kernel_size / 2)

        self.__residual_link = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        
        if in_channels != out_channels:
            self.__identity_link = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.__identity_link = nn.Identity()

        self.__after_addition = nn.ReLU()
        
        
    def forward(self, input: Tensor) -> Tensor:
        residual = self.__residual_link(input)
        identity = self.__identity_link(input)

        return self.__after_addition(identity + residual)



class Res1d18(nn.Module):
    def __init__(self, conf: Conf):
        super(Res1d18, self).__init__()

        # TODO fixed for binary classification
        assert conf.getHP('num_class') == 2
        num_class = 1

        self.__threshold = conf.getHP('threshold')

        in_channels = conf.getHP('num_input_channels')
        feature_channels = 64
        feature_kernel_size = 7
        feature_padding = int(feature_kernel_size / 2)
        feature_stride = 2
        pool_kernel_size = 3

        self.__feature = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=feature_channels, kernel_size=feature_kernel_size, stride=feature_stride, padding=feature_padding),
            nn.BatchNorm1d(feature_channels),
            nn.ReLU(),
        )

        self.__cnn = nn.Sequential(
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=feature_stride),
            _Res1d18Block(in_channels=feature_channels, out_channels=feature_channels),
            _Res1d18Block(in_channels=feature_channels, out_channels=feature_channels),
            
            _Res1d18Block(in_channels=feature_channels, out_channels=feature_channels * 2),
            _Res1d18Block(in_channels=feature_channels * 2, out_channels=feature_channels * 2),

            _Res1d18Block(in_channels=feature_channels * 2, out_channels=feature_channels * 4),
            _Res1d18Block(in_channels=feature_channels * 4, out_channels=feature_channels * 4),

            _Res1d18Block(in_channels=feature_channels * 4, out_channels=feature_channels * 8),
            _Res1d18Block(in_channels=feature_channels * 8, out_channels=feature_channels * 8),
        )

        self.__pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Squeeze()
        )

        self.__infer = nn.Sequential(
            nn.Linear(feature_channels * 8, num_class),

            # whether Softmax is necessary for binary classification
            # nn.Softmax(dim=1)

            Squeeze(),
            nn.Sigmoid()
        )


    def forward(self, input: Tensor, auxiliary_input: Tensor = None, refine: bool = False, detach: bool = False) -> Tensor:
        return self.__infer(self.__pool(self.__cnn(self.__feature(input))))


    def infer(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            predictions = self.forward(input).detach().cpu().numpy()

        return discretize(predictions, self.__threshold)
