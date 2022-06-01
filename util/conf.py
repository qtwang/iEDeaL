# coding = utf-8

import os
import json
import copy
from pathlib import Path

import numpy as np


class Conf:
    def __init__(self, conf_filepath: str = None, query_only: bool = False) -> None:
        self.query_only = query_only

        self.settings = {}
        
        self.defaults = {
            'model': 'resnet',
            
            'device': 'cuda',

            'checkpoint_filename': 'model.pickle',
            'checkpoint_filepath': None,
            'checkpoint_folderpath': None,

            'log_filename': 'train.log',
            'log_filepath': None,

            'default_conf_filename': 'conf.json',
            'conf_filepath': None,

            'num_epoch': 100,

            'train_positive_samples': None,
            'train_negative_samples': None,
            'validate_positive_samples': None,
            'validate_negative_samples': None,

            # 'size_train': 200000,
            # 'size_val': 10000,
            'batch_size': 256,

            'dim_series': 768,
            'num_input_channels': 5,
            'num_class': 2,
            'size_kernel': 3,
            'num_resblock': 7,
            'dim_latent': 768,
            'num_latent_channels': 256,
            'num_output_channels': 256,

            # 'dim_fully_connected': 256,
            
            'resblock_pre_activation': True,

            'dilation_type': 'exponential',
            'dilation_cons': 1,
            'dilation_base': 1,
            'dilation_slope': 1,
 
            'activation_conv': 'relu',
            'relu_slope': 1e-2,
            'activation_linear': 'lecuntanh',

            'model_init': 'lsuv',
            'lsuv_size': 'default', # -1: to decide according to train size, default: batch_size
            'lsuv_mean': 0,
            'lsuv_std': 1.,
            'lsuv_std_tol': 0.1,
            'lsuv_maxiter': 10,
            'lsuv_ortho': True,

            # 'orth_regularizer': 'none',
            # 'srip_mode': 'linear',
            # 'srip_cons': 0.1,
            # 'srip_max': 5e-4,
            # 'srip_min': 0,

            'layernorm_type': 'layernorm',
            'layernorm_elementwise_affine': True,

            'weightnorm_type': 'none',
            'weightnorm_dim': 0,

            'normalize_type': 'none',
            'mu': 0,
            'sigma': 1,

            'optim_type': 'sgd',
            'momentum': 0.9,
            'lr_mode': 'linear', 
            'lr_cons': 1e-3,
            'lr_max': 1e-3,
            'lr_min': 1e-6,
            'lr_everyk': 2,
            'lr_ebase': 0.9,
            'lr_steps': {60, 85},
            'lr_stepdevider': 10,
            'lr_warmup': 0.01,

            'wd_mode': 'fix', 
            'wd_cons': 1e-4,
            'wd_max': 1e-4,
            'wd_min': 1e-8,

            'adanorm_k': 1 / 10,
            'adanorm_scale': 2.,

            'loss_function': 'ce',
            
            # 'ce_weights': 'cvpr19',
            'num_samples_method': 'cvpr19',
            'sample_method': 'random',
            'sample_epoch': 1,

            'f_beta': 1,
            'f_alpha': 1,
            
            'num_training_negative_samples': -1,

            'threshold': 0.5,
            'float32_epsilon': 1e-7,

            'refine_loss_function': 'sf',
            'refine_extra_iterations': 1,
            # 'refine_bpfilter_iterations': {1}, # starting from 1, i.e., {1, 2, 3}, or None
            'refine_bpfilter_iterations': None,

            'warmup': True,
            'warmup_epochs': 3,

            'sampling_period': 1,

            'torch_rdseed': 1997,
            'cuda_rdseed': 1229,

            'early_stop_tracebacks': 5,
            'earlystop_patience': 3,
            'earlystop_epsilon': 1e-5,
            'earlystop_threshold_vfbeta': 0,

            'imbalance_expanding': 'none',
            'increxp_epochbase': 1.5,
            'increxp_numnegbase': 2,

            'clip_grad': 'none',
            'max_norm': 1,
            'norm_type': 2,
            'clip_value': 1,
            'clip_grad_warmup': True,

            'wce_pw': 'default',
            'wce_nw': 'default',

            'focal': False,
            'focal_gamma_p': 0.5,
            'focal_gamma_n': 0.5,

            'focalloss_gamma': 2,

            'dropout': True,
            'dropout_p': 0.5,

            'debug': False,

            'inception_bottleneck_channels': 64,
            'inception_kernel_sizes': [1, 3, 5, 9, 17],
            'num_inception_blocks': 5,
        }

        self.legals = {
            'device': {'cpu', 'cuda'},

            'model': {'resnet', 'resnetfr', 'res1d18', 'incept'},

            'checkpoint_mode': {'last', 'everyk', 'none'},

            'dilation_type': {'exponential', 'linear', 'fixed'},

            'weightnorm_type': {'weightnorm', 'none'},
            'layernorm_type': {'layernorm', 'adanorm', 'none'},

            'activation_conv': {'relu', 'leakyrelu', 'tanh', 'lecuntanh'},
            'activation_linear': {'relu', 'leakyrelu', 'tanh', 'lecuntanh'},

            'lr_mode': {'linear', 'fix', 'exponentially', 'exponentiallyhalve', 'plateauhalve', 'step'},
            'wd_mode': {'linear', 'fix'},
            
            'orth_regularizer': {'srip', 'none'},
            'srip_mode': {'linear', 'fix'},

            'model_init': {'lsuv', 'default'},

            'optim_type': {'sgd'},

            # wce by normalized 1 / effective num.: CVPR19 Class-Balanced Loss Based on Effective Number of Samples
            # normalization: Sigma_i^C alpha_i = C, i.e., alpha_p + alpha_n = 2
            # sf: arXiv21 A surrogate loss function for optimization of $F_\beta$ score in binary classification with imbalanced data 
            'loss_function': {'ce', 'wce', 'sf', 'sasu', 'focal', 'dice'},

            'num_samples_method': {'none', 'balanced', 'cvpr19'},
            'sample_method': {'random', 'latent'},

            'normalize_type': {'data', 'input', 'none'},
            'imbalance_expanding': {'exponential', 'none'},
            'clip_grad': {'norm', 'value', 'none'},

            'earlystop_type': {'continuous', 'mean', 'none'},
            'earlystop_target': {'tloss', 'tfbeta', 'vfbeta'},
        }

        self.useless_keys = {
            'not2tune'
        }

        if conf_filepath is not None:
            self.updateConf(conf_filepath)


    def getHP(self, key: str):
        if key in self.settings:
            return self.settings[key]
        
        if key in self.defaults:
            return self.defaults[key]
        
        raise ValueError('hyperparmeter {} doesn\'t exist'.format(key))


    def setHP(self, key: str, value):
        self.settings[key] = value


    def updateConf(self, config):
        if type(config) is not dict:
            assert type(config) is str and os.path.isfile(config)

            with open(config, 'r') as fin:
                config = json.load(fin)

        for key, value in config.items():
            if key not in self.useless_keys:
                if key not in self.legals or value in self.legals[key]:
                    self.setHP(key, value) 
                else:
                    raise ValueError('invalid hyperparameter {:s}: {:s}'.format(key, str(value)))


    def setup(self):
        # assert type(self.getHP('train_positive_samples')) is np.ndarray and type(self.getHP('train_negative_samples')) is np.ndarray
        # assert type(self.getHP('validate_positive_samples')) is np.ndarray and type(self.getHP('validate_negative_samples')) is np.ndarray

        log_filepath = self.getHP('log_filepath')
        assert log_filepath is not None and type(log_filepath) is str

        if not (os.path.exists(log_filepath) and os.path.isfile(log_filepath)):
            assert Path(log_filepath).parent.absolute().exists()

        if self.getHP('model_init') == 'lsuv' and self.getHP('lsuv_size') == 'default':
            self.setHP('lsuv_size', self.getHP('batch_size'))

        assert not (self.getHP('model') == 'resnetfr' and self.getHP('loss_function') == 'sasu')

        # assume loss_function, sample_method, sample_epoch are compatible
        # if self.getHP('loss_function') == 'sf' or self.getHP('loss_function') == 'sasu':
        #     assert self.getHP('f_beta') > 0 and self.getHP('f_alpha') >= 1

        if self.getHP('checkpoint_filepath') is None:
            assert self.getHP('checkpoint_folderpath') is not None and type(self.getHP('checkpoint_folderpath')) is str 

            self.setHP('checkpoint_filepath', os.path.join(self.getHP('checkpoint_folderpath'), self.getHP('checkpoint_filename')))

        if self.getHP('imbalance_expanding') != 'none':
            assert 1 < self.getHP('increxp_epochbase') <= self.getHP('increxp_numnegbase')

        if self.getHP('loss_function') == 'focal':
            assert not self.getHP('focal')


    def dumpConf(self, filepath: str = None) -> None:
        if filepath is None:
            conf_filepath = self.getHP('conf_filepath')
            assert conf_filepath is not None and type(conf_filepath) is str

            filepath = conf_filepath

        with open(filepath, 'w') as fout:
            self.settings_copy = copy.deepcopy(self.settings)
            
            for key2check in ['train_positive_samples', 'train_negative_samples', 'validate_positive_samples', 'validate_negative_samples']:
                if self.settings_copy[key2check] is not None and self.settings_copy[key2check] is not str:
                    self.settings_copy[key2check] = str(self.settings_copy[key2check].shape)

            json.dump({'settings': self.settings_copy, 'defaults': self.defaults}, fout, sort_keys=True, indent=4)
