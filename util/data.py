# coding = utf-8

from logging import raiseExceptions
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class Samples(Dataset):
    def __init__(self, samples):
        super(Samples, self).__init__()

        assert type(samples) is Tensor
        assert len(samples.shape) == 3

        self.__len = samples.shape[0]

        self.__samples = samples

        
    def __len__(self):
        return self.__len
    
    
    def __getitem__(self, indices):
        return self.__samples[indices]
    

class SamplesLabels(Dataset):
    def __init__(self, samples, labels):
        super(SamplesLabels, self).__init__()

        assert type(samples) is Tensor and type(labels) is Tensor
        assert len(samples.shape) == 3 and len(labels.shape) == 1
        assert samples.shape[0] == labels.shape[0]

        self.__len = samples.shape[0]

        self.__samples = samples
        self.__labels = labels

        
    def __len__(self):
        return self.__len
    
    
    def __getitem__(self, indices):
        return self.__samples[indices], self.__labels[indices]
    

class SamplesLabelsWeights(Dataset):
    def __init__(self, samples, labels, weights = None):
        super(SamplesLabelsWeights, self).__init__()
        
        assert type(samples) is Tensor and type(labels) is Tensor
        assert len(samples.shape) == 3 and len(labels.shape) == 1
        assert samples.shape[0] == labels.shape[0]

        self.__len = samples.shape[0]
        
        self.__samples = samples
        self.__labels = labels

        self.__weights = None

        if weights is not None:
            assert type(weights) is Tensor and len(weights.shape) == 1 and samples.shape[0] == weights.shape[0]
            
            self.__weights = weights
        else:
            self.__weights_filling = float('inf')

        
    def __len__(self):
        return self.__len
    
    
    def __getitem__(self, indices):
        # TODO verify this in source code
        # assert type(indices) == int

        if self.__weights is not None:
            return self.__samples[indices], self.__labels[indices], self.__weights[indices]
        else:
            return self.__samples[indices], self.__labels[indices], self.__weights_filling
    

def normalize(values: np.ndarray, axis = -1, mu: np.float32 = 0, sigma: np.float32 = 1, local: bool = True, epsilon: np.float32 = 1e-6):
    # TODO implement for common cases
    assert local
    assert axis == -1
    assert len(values.shape) == 3

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            # TODO not taking into consideration the performance
            local_mu = np.mean(values[i, j])
            local_sigma = np.sqrt(np.var(values[i, j]))

            if local_sigma <= epsilon:
                values[i, j] = mu + np.zeros_like(values[i, j])
            else:
                values[i, j] = mu + sigma * (values[i, j] - local_mu) / local_sigma
    
    return values
