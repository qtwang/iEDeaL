# coding = utf-8

import torch
import numpy as np
from torch import nn, Tensor

from model.commons import Mean, Sum


# A Surrogate Loss Function for Optimization of Fβ Score in Binary Classiﬁcation with Imbalanced Data
# Namgil Lee, Heejung Yang, Hojin Yoo
# https://arxiv.org/abs/2104.01459, [v1] Sat, 3 Apr 2021
class FbetaLoss(nn.Module):
    def __init__(self, beta: torch.float32 = 1., alpha: torch.float32 = 1., p2n: torch.float32 = 1., reduction: str = 'mean', device: str = 'cpu', epsilon=1e-7):
        super(FbetaLoss, self).__init__()

        self.epsilon = epsilon

        self.reset(beta, alpha, p2n, self.epsilon)

        if reduction == 'mean':
            self.__reduce = Mean()
        elif reduction == 'sum':
            self.__reduce = Sum()
        elif reduction == 'none':
            self.__reduce = nn.Identity()
        else:
            raise ValueError('invalid Fbeta reduction type: {:s}'.format(reduction))

    
    def reset(self, beta: torch.float32 = 1., alpha: torch.float32 = 1., p2n: torch.float32 = 1., margin: torch.float32 = 1e-3):
        self.__negative_offset: torch.float32 = p2n

        if np.abs(beta - 1) > margin:
            self.__negative_offset *= (beta ** 2)

        if np.abs(alpha - 1) > margin:
            self.__negative_offset /= alpha


    def forward(self, predictions: Tensor, targets: Tensor, weights: Tensor = None) -> Tensor:
        # assert weights.shape[0] == 1 and torch.any(torch.isinf(weights)) 

        # TODO numerical stability?
        lf = (1 - targets) * torch.log(self.__negative_offset + predictions) - targets * torch.log(predictions + self.epsilon)

        if weights is not None:
            if not torch.any(torch.isinf(weights)):
                lf *= weights

        return self.__reduce(lf)


class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()

        self.__loss = nn.BCELoss(weight=None, reduction=reduction)


    def forward(self, predictions: Tensor, targets: Tensor, weights: Tensor = None) -> Tensor:
        if weights is not None:
            if not torch.any(torch.isinf(weights)):
                self.__loss.weight = weights

        # Assertion `input_val >= zero && input_val <= one` failed. 
        return self.__loss(predictions, targets)
