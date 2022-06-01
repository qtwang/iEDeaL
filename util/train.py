# coding = utf-8

import imp
import sys
import os
import gc
import logging  
from pathlib import Path
from timeit import default_timer as timer

import torch
import numpy as np
# from ray import tune
from torch import nn, optim
from torch.utils.data import DataLoader
from lib.mpmath.functions.functions import re

from model.build import getModel
from model.initialize import LSUVinit
from model.loss import BCELoss, FbetaLoss
from util.conf import Conf
from util.data import Samples, SamplesLabelsWeights, normalize
from util.evaluate import Fbeta
from util.commons import discretize
from util.sample import Sampler
from lib.mpmath import mp
from util import autograd_hacks  


# class EEGTrainable(tune.Trainable):
class EEGTrainable():
    def __init__(self, config = None, train: bool = True):
        if config is not None:
            if type(config) is Conf:
                self.__conf = config
            elif type(config) is dict:
                # for ray tune
                self.__conf = config['not2tune']

                assert type(self.__conf) is Conf

                self.__conf.updateConf(config)
            else:
                raise ValueError('invalid config type: {:s}'.format(type(config)))
        else:
            self.__conf = Conf()

        self.__debug = self.__conf.getHP('debug')

        self.__device = self.__conf.getHP('device')
        self.__batch_size = self.__conf.getHP('batch_size')

        self.__threshold = self.__conf.getHP('threshold')
        
        self.model = getModel(self.__conf)
        self.model.to(self.__device)

        if train:
            self.__conf.setup()
            self.__setup()
        else:
            checkpoint_filepath = self.__conf.getHP('checkpoint_filepath')
            assert os.path.isfile(checkpoint_filepath)
            self.__load(checkpoint_filepath)

        gc.collect()
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def __load(self, checkpoint_filepath: str):
        self.model.load_state_dict(torch.load(checkpoint_filepath))


    def __setup(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(self.__class__.__name__)

        filehandler = logging.FileHandler(self.__conf.getHP('log_filepath'), 'a+')
        formatter = logging.Formatter('%(asctime)s,%(msecs)d %(levelname).3s [%(filename)s:%(lineno)d] %(message)s', datefmt='%m/%d/%Y:%I:%M:%S')
        filehandler.setFormatter(formatter)
        # filehandler.setLevel(logging.DEBUG)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.addHandler(filehandler)

        self.max_epoch = self.__conf.getHP('num_epoch')
        self.iteration = 0
        
        self.ready4stop_epoch = self.max_epoch
        self.earlystop_wait_epochs = self.__conf.getHP('early_stop_tracebacks')

        if type(self.__conf.getHP('train_positive_samples')) is str:
            assert os.path.isfile(self.__conf.getHP('train_positive_samples'))
            train_positive_samples = np.fromfile(self.__conf.getHP('train_positive_samples'), dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')])
        elif type(self.__conf.getHP('train_positive_samples')) is list:
            train_positive_samples = []

            for tp_sample_path in self.__conf.getHP('train_positive_samples'):
                assert os.path.isfile(tp_sample_path)
                train_positive_samples.append(np.fromfile(tp_sample_path, dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')]))

            train_positive_samples = np.concatenate(train_positive_samples)
        else:
            assert type(self.__conf.getHP('train_positive_samples')) is np.ndarray
            train_positive_samples = self.__conf.getHP('train_positive_samples')

        if type(self.__conf.getHP('train_negative_samples')) is str:
            assert os.path.isfile(self.__conf.getHP('train_negative_samples'))
            train_negative_samples = np.fromfile(self.__conf.getHP('train_negative_samples'), dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')])
        elif type(self.__conf.getHP('train_negative_samples')) is list:
            train_negative_samples = []

            for tn_sample_path in self.__conf.getHP('train_negative_samples'):
                assert os.path.isfile(tn_sample_path)
                train_negative_samples.append(np.fromfile(tn_sample_path, dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')]))

            train_negative_samples = np.concatenate(train_negative_samples)
        else:
            assert type(self.__conf.getHP('train_negative_samples')) is np.ndarray
            train_negative_samples = self.__conf.getHP('train_negative_samples')

        if type(self.__conf.getHP('validate_positive_samples')) is str:
            assert os.path.isfile(self.__conf.getHP('validate_positive_samples'))
            validate_positive_samples = np.fromfile(self.__conf.getHP('validate_positive_samples'), dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')])
        elif type(self.__conf.getHP('validate_positive_samples')) is list:
            validate_positive_samples = []

            for vp_sample_path in self.__conf.getHP('validate_positive_samples'):
                assert os.path.isfile(vp_sample_path)
                validate_positive_samples.append(np.fromfile(vp_sample_path, dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')]))

            validate_positive_samples = np.concatenate(validate_positive_samples)
        else:
            assert type(self.__conf.getHP('validate_positive_samples')) is np.ndarray
            validate_positive_samples = self.__conf.getHP('validate_positive_samples')

        if type(self.__conf.getHP('validate_negative_samples')) is str:
            assert os.path.isfile(self.__conf.getHP('validate_negative_samples'))
            validate_negative_samples = np.fromfile(self.__conf.getHP('validate_negative_samples'), dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')])
        elif type(self.__conf.getHP('validate_negative_samples')) is list:
            validate_negative_samples = []

            for vn_sample_path in self.__conf.getHP('validate_negative_samples'):
                assert os.path.isfile(vn_sample_path)
                validate_negative_samples.append(np.fromfile(vn_sample_path, dtype=np.float32).reshape([-1, self.__conf.getHP('num_input_channels'), self.__conf.getHP('dim_series')]))

            validate_negative_samples = np.concatenate(validate_negative_samples)
        else:
            assert type(self.__conf.getHP('validate_negative_samples')) is np.ndarray
            validate_negative_samples = self.__conf.getHP('validate_negative_samples')

        assert train_positive_samples is not None and train_negative_samples is not None
        assert validate_positive_samples is not None and validate_negative_samples is not None

        train_positive_samples = np.asarray(train_positive_samples)
        train_negative_samples = np.asarray(train_negative_samples)
        validate_positive_samples = np.asarray(validate_positive_samples)
        validate_negative_samples = np.asarray(validate_negative_samples)

        assert len(train_positive_samples.shape) == len(train_negative_samples.shape) == 3
        assert len(validate_positive_samples.shape) == len(validate_negative_samples.shape) == 3

        assert 0 < train_positive_samples.shape[0] <= train_negative_samples.shape[0]
        assert 0 < validate_positive_samples.shape[0] <= validate_negative_samples.shape[0]

        assert train_positive_samples.shape[1] == train_negative_samples.shape[1] == self.__conf.getHP('num_input_channels')
        assert validate_positive_samples.shape[1] == validate_negative_samples.shape[1] == self.__conf.getHP('num_input_channels')

        assert train_positive_samples.shape[2] == train_negative_samples.shape[2] == self.__conf.getHP('dim_series')
        assert validate_positive_samples.shape[2] == validate_negative_samples.shape[2] == self.__conf.getHP('dim_series')

        assert self.__conf.getHP('num_class') == 2

        if self.__conf.getHP('num_samples_method') == 'balanced':
            self.__num_train_negative_samples = train_positive_samples.shape[0]
        elif self.__conf.getHP('num_samples_method') == 'cvpr19':
            N = mp.mpf(train_positive_samples.shape[0] + train_negative_samples.shape[0])
            beta = (N - mp.mpf(1.0)) / N
            effective = (mp.mpf(1.0) - beta ** mp.mpf(train_negative_samples.shape[0])) / (mp.mpf(1.0) - beta)
            self.__num_train_negative_samples = int(np.rint(np.float64(effective)))
        else:
            self.__num_train_negative_samples = train_negative_samples.shape[0]
        
        self.__conf.setHP('num_training_negative_samples', self.__num_train_negative_samples)
        assert train_positive_samples.shape[0] <= self.__num_train_negative_samples <= train_negative_samples.shape[0]

        self.__loss_function = self.__conf.getHP('loss_function')
        if self.__loss_function == 'sasu' or self.__loss_function == 'sf':
            assert self.__conf.getHP('f_beta') > 0

            if self.__loss_function == 'sasu':
                self.__conf.setHP('f_alpha', train_negative_samples.shape[0] / self.__num_train_negative_samples)

        if self.__conf.getHP('normalize') == 'data':
            mu = self.__conf.getHP('mu')
            sigma = self.__conf.getHP('sigma')

            train_positive_samples = normalize(train_positive_samples, mu=mu, sigma=sigma)
            train_negative_samples = normalize(train_negative_samples, mu=mu, sigma=sigma)
            validate_positive_samples = normalize(validate_positive_samples, mu=mu, sigma=sigma)
            validate_negative_samples = normalize(validate_negative_samples, mu=mu, sigma=sigma)

        self.__num_train_positive_samples = train_positive_samples.shape[0]
        self.__num_train_negative_samples = self.__conf.getHP('num_training_negative_samples')
        self.__total_num_train_negative_samples = train_negative_samples.shape[0]
        num_train_samples = self.__num_train_positive_samples + self.__num_train_negative_samples

        self.__train_positive_samples = torch.from_numpy(np.asarray(train_positive_samples)).to(self.__device)
        self.__train_negative_samples = torch.from_numpy(np.asarray(train_negative_samples)).to(self.__device)
        self.__all_train_samples = torch.cat([self.__train_positive_samples, self.__train_negative_samples])

        self.__train_labels = torch.zeros(num_train_samples, device=self.__device, requires_grad=False)
        self.__train_labels[: self.__num_train_positive_samples] = 1

        self.__train_weights = None

        if self.__debug:
            self.__all_train_labels_debug = np.zeros(self.__all_train_samples.shape[0], dtype=int)
            self.__all_train_labels_debug[: self.__num_train_positive_samples] = 1

            self.__all_train_dataloader = DataLoader(Samples(self.__all_train_samples), batch_size=self.__batch_size, shuffle=False)

        self.__focal = False
        if self.__loss_function == 'sasu':
            self.__focal = self.__conf.getHP('focal')

            if self.__focal:
                self.__focal_gamma_p = self.__conf.getHP('focal_gamma_p')
                self.__focal_gamma_n = self.__conf.getHP('focal_gamma_n')
    
        if self.__loss_function == 'wce':
            self.__train_weights = torch.ones(num_train_samples, device=self.__device, requires_grad=False)

            if self.__conf.getHP('wce_pw') != 'default':
                pweight = self.__conf.getHP('wce_pw')

                if self.__conf.getHP('wce_nw') != 'default':
                    nweight = self.__conf.getHP('wce_nw')
                else:
                    nweight = 1 - pweight
            else:
                unnorm_pweight = 1 / self.__num_train_positive_samples
                unnorm_nweight = 1 / self.__num_train_negative_samples

                pweight = unnorm_pweight / (unnorm_pweight + unnorm_nweight)
                nweight = unnorm_nweight / (unnorm_pweight + unnorm_nweight)

            self.__train_weights[: self.__num_train_positive_samples] = pweight
            self.__train_weights[self.__num_train_positive_samples: ] = nweight

            self.logger.info('wp = {:.3f}, wn = {:.3f}'.format(self.__train_weights[0], self.__train_weights[-1]))
        elif self.__loss_function == 'focal':
            assert not self.__focal

            self.__focal_gamma = self.__conf.getHP('focalloss_gamma')
            self.__focal_gamma_p = self.__focal_gamma
            self.__focal_gamma_n = self.__focal_gamma
            
        if self.__loss_function == 'focal' or self.__focal:
            self.logger.info('focal_gamma_p/n = {:.2f}/{:.2f}'.format(self.__focal_gamma_p, self.__focal_gamma_n))

        if self.__num_train_negative_samples == self.__total_num_train_negative_samples:
            self.__train_dataloader = DataLoader(SamplesLabelsWeights(self.__all_train_samples, 
                                                                      self.__train_labels, 
                                                                      self.__train_weights), 
                                                 batch_size=self.__batch_size, shuffle=True)

            self.logger.info('ntp = {:d}, ntn = {:d}'.format(self.__num_train_positive_samples, self.__num_train_negative_samples))

            self.ready4stop_epoch = self.earlystop_wait_epochs
            self.logger.info('ready to early stop from t{:d}'.format(self.ready4stop_epoch))

        validate_positive_samples = torch.from_numpy(np.asarray(validate_positive_samples)).to(self.__device)
        validate_negative_samples = torch.from_numpy(np.asarray(validate_negative_samples)).to(self.__device)

        validate_samples = torch.cat([validate_positive_samples, validate_negative_samples])
        self.logger.info('nvp = {:d}, nvn = {:d}'.format(validate_positive_samples.shape[0], validate_negative_samples.shape[0]))

        self.__validate_labels = np.zeros(validate_positive_samples.shape[0] + validate_negative_samples.shape[0], dtype=int)
        self.__validate_labels[: validate_positive_samples.shape[0]] = 1
        
        self.__validate_dataloader = DataLoader(Samples(validate_samples), batch_size=self.__batch_size, shuffle=False)

        self.imbalance_expanding = self.__conf.getHP('imbalance_expanding')
        if self.imbalance_expanding == 'exponential':
            assert self.__num_train_negative_samples > self.__num_train_positive_samples

            self.expanding_finished = False
            
            self.current_imbalance_ratio = 1
            self.current_epch2expand = 1
            self.next_expansion = 1

            self.increxp_epochbase = self.__conf.getHP('increxp_epochbase')
            self.increxp_numnegbase = self.__conf.getHP('increxp_numnegbase')

            self.current_num_train_negative_samples = int(self.current_imbalance_ratio * self.__num_train_positive_samples)

        self.clip_grad = self.__conf.getHP('clip_grad')
        if self.clip_grad != 'none':
            self.clip_grad_warmup = self.__conf.getHP('clip_grad_warmup')

            if self.clip_grad == 'norm':
                self.max_norm = self.__conf.getHP('max_norm')
                self.norm_type = self.__conf.getHP('norm_type')
            elif self.clip_grad == 'value':
                self.clip_value = self.__conf.getHP('clip_value')

        if self.__conf.getHP('model') == 'resnetfr':
            self.__refine_extra_iterations = self.__conf.getHP('refine_extra_iterations')
            self.__refine_bpfilter_iterations = self.__conf.getHP('refine_bpfilter_iterations')

            self.__trainF = self.__trainFR
            self.__loss_refine = self.__getLoss(self.__conf.getHP('refine_loss_function'))

            self.__all_train_dataloader = DataLoader(Samples(self.__all_train_samples), batch_size=self.__batch_size, shuffle=False)
            self.__all_train_labels = self.__train_labels

            self.__all_train_weights = self.__train_weights

            if self.__num_train_negative_samples != self.__total_num_train_negative_samples:
                self.__all_train_labels = torch.zeros(self.__num_train_positive_samples + self.__total_num_train_negative_samples, device=self.__device, requires_grad=False)
                self.__all_train_labels[: self.__num_train_positive_samples] = 1

                if self.__loss_function == 'ce':
                    self.__all_train_weights = torch.ones(self.__num_train_positive_samples + self.__total_num_train_negative_samples, device=self.__device, requires_grad=False)

                    if self.__conf.getHP('ce_weights') == 'cvpr19':
                        pweight = 1 / self.__num_train_positive_samples
                        nweight = 1 / self.__num_train_negative_samples

                        self.__all_train_weights[: self.__num_train_positive_samples] = 2 * pweight / (pweight + nweight)
                        self.__all_train_weights[self.__num_train_positive_samples: ] = 2 * nweight / (pweight + nweight)
    
            self.__all_train_labels_numpy = self.__all_train_labels.cpu().numpy()
        else:
            self.__trainF = self.__train
        
        self.__optimizer = self.__getOptimizer()
        self.__loss = self.__getLoss(self.__loss_function)

        self.__initModel()
        # autograd_hacks.add_hooks(self.model)        

        if self.__conf.getHP('warmup'):
            if self.__conf.getHP('model') != 'resnetfr':
                self.__all_train_dataloader = DataLoader(Samples(self.__all_train_samples), batch_size=self.__batch_size, shuffle=False)
                self.__all_train_labels = self.__train_labels

                if self.__num_train_negative_samples != self.__total_num_train_negative_samples:
                    self.__all_train_labels = torch.zeros(self.__num_train_positive_samples + self.__total_num_train_negative_samples, device=self.__device, requires_grad=False)
                    self.__all_train_labels[: self.__num_train_positive_samples] = 1

                self.__all_train_labels_numpy = self.__all_train_labels.cpu().numpy()

            self.__warmupModel()

        self.sampler = None
        if self.__num_train_negative_samples < self.__total_num_train_negative_samples and self.__conf.getHP('sample_method') == 'latent':
            self.sampler = Sampler(self.__conf)


    def step(self):
        start = timer()

        self.__adjustLR()
        self.__adjustWD()

        train_loss, tfbeta, tprecision, trecall = self.__trainF()

        if type(train_loss) is tuple:
            self.logger.info('t{:d} in {:.3f}s: lf={:.4f}, lr={:.4f}, f{:.1f}={:.4f}, p={:.4f}, r={:.4f}'.format(
                self.iteration, timer() - start, train_loss[0], train_loss[1], self.__conf.getHP('f_beta'), tfbeta, tprecision, trecall))
        else:
            self.logger.info('t{:d} in {:.3f}s: l={:.4f}, f{:.1f}={:.4f}, p={:.4f}, r={:.4f}'.format(
                self.iteration, timer() - start, train_loss, self.__conf.getHP('f_beta'), tfbeta, tprecision, trecall))
        
        start = timer()

        vfbeta, vprecision, vrecall = self.__validate()

        self.logger.info('v{:d} in {:.3f}s: f{:.1f}={:.4f}, p={:.4f}, r={:.4f}'.format(
            self.iteration, timer() - start, self.__conf.getHP('f_beta'), vfbeta, vprecision, vrecall))

        self.iteration += 1

        return train_loss, tfbeta, tprecision, trecall, vfbeta, vprecision, vrecall


    def __train(self) -> None:
        losses = []

        predictions = []
        targets = []

        if self.__num_train_negative_samples != self.__total_num_train_negative_samples:
            start = timer()

            negative_indices = None

            if self.iteration == 0:
                negative_indices = 'random'
            elif self.iteration % self.__conf.getHP('sampling_period') == 0:
                if self.sampler is None:
                    negative_indices = 'random'
                else:
                    negative_indices = self.sampler.sample(self.__getLatent()) + self.__num_train_positive_samples

            if negative_indices is not None:
                if self.imbalance_expanding != 'none':
                    if not self.expanding_finished:
                        expanded = False

                        if self.imbalance_expanding == 'exponential':
                            if self.iteration >= self.current_epch2expand:
                                self.current_imbalance_ratio = int(np.floor(self.increxp_numnegbase ** self.next_expansion))

                                self.next_expansion += 1
                                self.current_epch2expand = int(np.floor(self.increxp_epochbase ** self.next_expansion))

                                expanded = True

                                self.logger.info('expansion {:d} at epoch {:d} to {:d}'.format(self.next_expansion - 1, self.iteration, self.current_imbalance_ratio))
                        else:
                            raise NotImplementedError('expansion {:s} not implemented yet'.format(self.imbalance_expanding))
                            
                        self.current_num_train_negative_samples = int(self.current_imbalance_ratio * self.__num_train_positive_samples)

                        if self.current_num_train_negative_samples >= self.__num_train_negative_samples:
                            if self.current_num_train_negative_samples > self.__num_train_negative_samples:
                                self.current_num_train_negative_samples = self.__num_train_negative_samples

                                self.logger.info('illegal expansion, rollback to {:.3f}'.format(self.current_num_train_negative_samples / self.__num_train_positive_samples))

                            self.expanding_finished = True
                            self.logger.info('expansion finished with {:.3f}'.format(self.current_num_train_negative_samples / self.__num_train_positive_samples))

                        if expanded and self.expanding_finished:
                            if self.next_expansion > 2:
                                epochs2wait = int(np.floor(self.increxp_epochbase ** (self.next_expansion - 2) * (self.increxp_epochbase - 1)))
                            else:
                                epochs2wait = self.earlystop_wait_epochs

                            if epochs2wait < self.earlystop_wait_epochs:
                                epochs2wait = self.earlystop_wait_epochs

                            self.ready4stop_epoch = self.iteration + epochs2wait
                            self.logger.info('early stop from t{:d}'.format(self.ready4stop_epoch))
                else:
                    self.current_num_train_negative_samples = self.__num_train_negative_samples

                if type(negative_indices) is str and negative_indices == 'random':
                    negative_indices = np.random.permutation(
                        np.arange(self.__num_train_positive_samples, 
                                  self.__num_train_positive_samples + self.__total_num_train_negative_samples)
                        )[: self.current_num_train_negative_samples]

                    sample_str = 'random'
                else:
                    sample_str = 'latent'

                indices = np.concatenate([np.arange(self.__num_train_positive_samples), negative_indices])

                if self.__train_weights is not None:
                    current_train_weights = self.__train_weights[: self.__num_train_positive_samples + self.current_num_train_negative_samples]
                else:
                    current_train_weights = self.__train_weights

                self.__train_dataloader = DataLoader(SamplesLabelsWeights(self.__all_train_samples[indices], 
                                                                          self.__train_labels[: self.__num_train_positive_samples + self.current_num_train_negative_samples], 
                                                                          current_train_weights), 
                                                     batch_size=self.__batch_size, shuffle=True)
            
                self.logger.info('sample ({:s}, {:d}/{:d}) t{:d} in {:.3f}s'.format(
                    sample_str, self.current_num_train_negative_samples, self.__total_num_train_negative_samples, self.iteration, timer() - start))

        for batch_samples, batch_labels, batch_weights in self.__train_dataloader:
            # autograd_hacks.clear_backprops(self.model)
            self.__optimizer.zero_grad()

            batch_predictions = self.model(batch_samples)

            if self.__loss_function == 'focal' or self.__focal:
                detached_predictions = batch_predictions.detach()
                focal_weights = batch_labels * (1 - detached_predictions) ** self.__focal_gamma_p + (1 - batch_labels) * detached_predictions ** self.__focal_gamma_n

                if not torch.any(torch.isinf(batch_weights)):
                    batch_weights *= focal_weights
                else:
                    batch_weights = focal_weights

            batch_loss = self.__loss(batch_predictions, batch_labels, batch_weights)

            batch_loss.backward()
            # autograd_hacks.compute_grad1(self.model)

            if self.clip_grad != 'none':
                if self.clip_grad == 'norm':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type, error_if_nonfinite=True)
                elif self.clip_grad == 'value':
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)

            self.__optimizer.step()

            losses.append(batch_loss.detach().item())
            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(batch_labels.clone().detach().cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        predictions = discretize(predictions, self.__threshold)

        targets = np.concatenate(targets, axis=0)

        fbeta, precision, recall = Fbeta(predictions, targets)

        if self.__debug:
            start = timer()
            predictions = []

            with torch.no_grad():
                for batch_samples in self.__all_train_dataloader:
                    predictions.append(self.model.infer(batch_samples))
            
            predictions = np.concatenate(predictions, axis=0)
            fbeta_debug, precision_debug, recall_debug = Fbeta(predictions, self.__all_train_labels_debug)

            self.logger.debug('t{:d}(debug) in {:.3f}s: f{:.1f}={:.4f}, p={:.4f}, r={:.4f}'.format(
                self.iteration, timer() - start, self.__conf.getHP('f_beta'), fbeta_debug, precision_debug, recall_debug))
        
        return np.mean(losses), fbeta, precision, recall

    
    def ready4stop(self):
        return self.iteration > self.ready4stop_epoch


    def __trainFR(self) -> None:
        train_loss, _, _, _ = self.__train()

        predictions = []

        with torch.no_grad():
            for batch_samples in self.__all_train_dataloader:
                predictions.append(self.model.infer(batch_samples))

        predictions = np.concatenate(predictions, axis=0)

        fbeta, precision, recall = Fbeta(predictions, self.__all_train_labels_numpy)

        filter_samples_indices = np.concatenate((np.asarray(range(self.__num_train_positive_samples)), 
                                                 np.squeeze(np.argwhere(predictions[self.__num_train_positive_samples: ]))))

        if self.__conf.getHP('refine_loss_function') != 'ce':
            self.__loss_refine.reset(p2n=self.__num_train_positive_samples / (np.sum(predictions) - self.__num_train_positive_samples))

        losses = []

        filter_weights = None
    
        filter_dataloader = DataLoader(SamplesLabelsWeights(self.__all_train_samples[filter_samples_indices], 
                                                            self.__all_train_labels[filter_samples_indices], 
                                                            filter_weights), 
                                                batch_size=self.__batch_size, shuffle=True)

        for epoch in range(1, self.__refine_extra_iterations + 1):
            detach = self.__refine_bpfilter_iterations is None or epoch not in self.__refine_bpfilter_iterations

            for batch_samples, batch_labels, batch_weights in filter_dataloader:
                self.__optimizer.zero_grad()

                batch_predictions = self.model(batch_samples, refine=True, detach=detach)

                batch_loss = self.__loss_refine(batch_predictions, batch_labels, batch_weights)

                batch_loss.backward()
                self.__optimizer.step()

                if epoch == self.__refine_extra_iterations:
                    losses.append(batch_loss.detach().item())
        
        return (train_loss, np.mean(losses)), fbeta, precision, recall


    def __getLatent(self) -> np.ndarray:
        latents = []

        for batch_samples in DataLoader(Samples(self.__train_negative_samples), batch_size=self.__batch_size, shuffle=False):
            self.model(batch_samples)
            latents.append(self.model.latent4label.detach().cpu().numpy())

        return np.concatenate(latents, axis=0)


    def __validate(self) -> None:
        predictions = []

        for batch_samples in self.__validate_dataloader:
            predictions.append(self.model.infer(batch_samples))

        predictions = np.concatenate(predictions, axis=0)

        return Fbeta(predictions, self.__validate_labels)


    def log_predictions(self):
        predictions = []

        with torch.no_grad():
            for batch_samples in self.__all_train_dataloader:
                predictions.append(self.model.infer(batch_samples))
        
        predictions = np.concatenate(predictions, axis=0)

        with np.printoptions(threshold=sys.maxsize):
            self.logger.debug('train predictions:\n{:s}'.format(np.array2string(predictions, separator=',')))
    
        predictions = []

        with torch.no_grad():
            for batch_samples in self.__validate_dataloader:
                predictions.append(self.model.infer(batch_samples))
        
        predictions = np.concatenate(predictions, axis=0)

        with np.printoptions(threshold=sys.maxsize):
            self.logger.debug('valid predictions:\n{:s}'.format(np.array2string(predictions, separator=',')))
    

    def predict(self, samples: np.ndarray) -> np.ndarray:
        dim_series = self.__conf.getHP('dim_series')
        num_channels = self.__conf.getHP('num_input_channels')

        samples = np.asarray(samples, dtype=np.float32)

        assert len(samples.shape) == 3 and samples.shape[1] == num_channels and samples.shape[2] == dim_series

        samples = torch.from_numpy(samples).to(self.__device)

        predictions = []

        for batch in DataLoader(samples, batch_size=self.__batch_size, shuffle=False):
            predictions.append(self.model.infer(batch))

        predictions = np.concatenate(predictions, axis=0)

        del samples
        gc.collect()
        torch.cuda.empty_cache()

        return predictions


    def cleanup(self, checkpoint: bool = True):
        self.logger.info('train finishes by {:d}'.format(self.iteration))

        if checkpoint:
            checkpoint_filepath = self.__conf.getHP('checkpoint_filepath')

            assert checkpoint_filepath is not None and type(checkpoint_filepath) is str 
            assert not os.path.exists(checkpoint_filepath) and Path(checkpoint_filepath).parent.absolute().exists()

            torch.save(self.model.state_dict(), checkpoint_filepath)
 
        del self.__all_train_samples
        del self.__train_dataloader
        
        del self.__validate_dataloader

        del self.model
        del self.__optimizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def __warmupModel(self) -> nn.Module:
        self.__adjustLR(self.__conf.getHP('lr_warmup'))

        warmup_loss = BCELoss()
        warmup_epochs = self.__conf.getHP('warmup_epochs')
        
        warmup_samples = torch.cat([
            self.__train_positive_samples, 
            self.__train_negative_samples[np.random.permutation(self.__num_train_negative_samples)][: self.__num_train_positive_samples],
        ])

        warmup_labels = self.__train_labels[: warmup_samples.shape[0]]

        warmup_dataloader = DataLoader(SamplesLabelsWeights(warmup_samples, warmup_labels), batch_size=self.__batch_size, shuffle=True)

        for warmup_epoch in range(warmup_epochs):
            start = timer()

            losses = []

            predictions = []
            targets = []

            for batch_samples, batch_labels, batch_weights in warmup_dataloader:
                # autograd_hacks.clear_backprops(self.model)
                self.__optimizer.zero_grad()

                batch_predictions = self.model(batch_samples)

                batch_loss = warmup_loss(batch_predictions, batch_labels, batch_weights)

                batch_loss.backward()
                # autograd_hacks.compute_grad1(self.model)
                
                if self.clip_grad != 'none' and self.clip_grad_warmup:
                    if self.clip_grad == 'norm':
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type, error_if_nonfinite=True)
                    elif self.clip_grad == 'value':
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)

                self.__optimizer.step()

                losses.append(batch_loss.detach().item())
                
                predictions.append(batch_predictions.detach().cpu().numpy())
                targets.append(batch_labels.clone().detach().cpu().numpy())

                if self.__conf.getHP('model') == 'resnetfr':
                    self.__optimizer.zero_grad()
                    
                    batch_predictions = self.model(batch_samples, refine=True, detach=True)

                    batch_loss = warmup_loss(batch_predictions, batch_labels, batch_weights)

                    batch_loss.backward()
                    self.__optimizer.step()
        
            predictions = np.concatenate(predictions, axis=0)
            predictions = discretize(predictions, self.__threshold)

            targets = np.concatenate(targets, axis=0)

            fbeta, precision, recall = Fbeta(predictions, targets)

            self.logger.info('w{:d} in {:.3f}s: l={:.4f}, f{:.1f}={:.4f}, p={:.4f}, r={:.4f}'.format(
                warmup_epoch, timer() - start, np.mean(losses), self.__conf.getHP('f_beta'), fbeta, precision, recall))

        start = timer()
        predictions = []

        with torch.no_grad():
            for batch_samples in self.__all_train_dataloader:
                predictions.append(self.model(batch_samples).detach().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        predictions = discretize(predictions, self.__threshold)

        fbeta, precision, recall = Fbeta(predictions, self.__all_train_labels_numpy)

        self.logger.info('t-1 in {:.3f}s: f{:.1f}={:.4f}, p={:.4f}, r={:.4f}'.format(
            timer() - start, self.__conf.getHP('f_beta'), fbeta, precision, recall))

        start = timer()
        predictions = []

        with torch.no_grad():
            for batch_samples in self.__validate_dataloader:
                predictions.append(self.model(batch_samples).detach().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        predictions = discretize(predictions, self.__threshold)

        fbeta, precision, recall = Fbeta(predictions, self.__validate_labels)

        self.logger.info('v-1 in {:.3f}s: f{:.1f}={:.4f}, p={:.4f}, r={:.4f}'.format(
            timer() - start, self.__conf.getHP('f_beta'), fbeta, precision, recall))
        

    def __initModel(self) -> nn.Module:
        if self.__conf.getHP('model_init') == 'lsuv':
            lsuv_size = self.__conf.getHP('lsuv_size')

            if lsuv_size < 0:
                lsuv_samples = torch.cat([
                    self.__train_positive_samples, 
                    self.__train_negative_samples[np.random.permutation(self.__num_train_negative_samples)][: self.__num_train_positive_samples],
                ])
            else:
                lsuv_samples = torch.cat([
                    self.__train_positive_samples[np.random.permutation(self.__num_train_positive_samples)][: int(lsuv_size / 2)], 
                    self.__train_negative_samples[np.random.permutation(self.__num_train_negative_samples)][: int(lsuv_size / 2)],
                ])
            
            lsuv_samples = lsuv_samples[np.random.permutation(lsuv_samples.shape[0])]

            self.model = LSUVinit(self.model, 
                                  lsuv_samples,
                                  needed_mean=self.__conf.getHP('lsuv_mean'), 
                                  needed_std=self.__conf.getHP('lsuv_std'), 
                                  std_tol=self.__conf.getHP('lsuv_std_tol'), 
                                  max_attempts=self.__conf.getHP('lsuv_maxiter'), 
                                  do_orthonorm=self.__conf.getHP('lsuv_ortho'),
                                  logger=self.logger)


    def __getLoss(self, loss_function) -> nn.Module:
        if loss_function == 'ce' or loss_function == 'wce' or loss_function == 'focal':
            return BCELoss()
        elif loss_function == 'sf' or loss_function == 'sasu':
            float32_epsilon = self.__conf.getHP('float32_epsilon')

            beta = self.__conf.getHP('f_beta')
            p2n = self.__num_train_positive_samples / self.__num_train_negative_samples

            alpha = self.__conf.getHP('f_alpha')

            if loss_function == 'sf':
                assert np.abs(alpha - 1) < float32_epsilon

            # return FbetaLoss(beta=beta, alpha=alpha, p2n=p2n, epsilon=float32_epsilon, reduction='none')
            return FbetaLoss(beta=beta, alpha=alpha, p2n=p2n, epsilon=float32_epsilon)
        elif loss_function == 'dice':
            raise NotImplementedError('loss function: {:s} not implemented'.format(loss_function))
        else:
            raise ValueError('invalid loss function: {:s}'.format(loss_function))


    def __getOptimizer(self) -> optim.Optimizer:
        if self.__conf.getHP('optim_type') == 'sgd':
            if self.__conf.getHP('lr_mode') == 'fix':
                initial_lr = self.__conf.getHP('lr_cons')
            else:
                initial_lr = self.__conf.getHP('lr_max')

            if self.__conf.getHP('wd_mode') == 'fix':
                initial_wd = self.__conf.getHP('wd_cons')
            else:
                initial_wd = self.__conf.getHP('wd_min')

            momentum = self.__conf.getHP('momentum')

            return optim.SGD(self.model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=initial_wd)

        raise ValueError('invalid optimizer name: {:s}'.format(self.__conf.getHP('optim_type')))


    def __adjustLR(self, new_lr = None) -> None:
        current_lr = None

        if new_lr == None:
            for param_group in self.__optimizer.param_groups:
                current_lr = param_group['lr']
                break

            if self.iteration == 0:
                if self.__conf.getHP('lr_mode') == 'fix':
                    new_lr = self.__conf.getHP('lr_cons')
                else:
                    new_lr = self.__conf.getHP('lr_max')
            else:
                new_lr = current_lr

                if self.__conf.getHP('lr_mode') == 'linear':
                    lr_max = self.__conf.getHP('lr_max')
                    lr_min = self.__conf.getHP('lr_min')

                    new_lr = lr_max - self.iteration * (lr_max - lr_min) / self.max_epoch
                elif self.__conf.getHP('lr_mode') == 'exponentiallyhalve':
                    lr_max = self.__conf.getHP('lr_max')
                    lr_min = self.__conf.getHP('lr_min')
                    
                    for i in range(1, 11):
                        if (self.max_epoch - self.iteration) * (2 ** i) == self.max_epoch:
                            new_lr = lr_max / (10 ** i)
                            break

                    if new_lr < lr_min:
                        new_lr = lr_min
                elif self.__conf.getHP('lr_mode') == 'exponentially':
                    lr_max = self.__conf.getHP('lr_max')
                    lr_min = self.__conf.getHP('lr_min')
                    lr_k = self.__conf.getHP('lr_everyk')
                    lr_ebase = self.__conf.getHP('lr_ebase')

                    lr_e = int(np.floor(self.iteration / lr_k))
                    new_lr = lr_max * (lr_ebase ** lr_e)

                    if new_lr < lr_min:
                        new_lr = lr_min
                elif self.__conf.getHP('lr_mode') == 'step':
                    if self.iteration in self.__conf.getHP('lr_steps'):
                        new_lr = self.__conf.getHP('lr_max')
                        lr_stepdevider = self.__conf.getHP('lr_stepdevider')

                        for step in self.__conf.getHP('lr_steps'):
                            if self.iteration >= step:
                                new_lr /= lr_stepdevider
                elif self.__conf.getHP('lr_mode') == 'plateauhalve':
                    raise ValueError('plateauhalve is not yet supported')

        if new_lr is not None:
            if current_lr is not None and current_lr != new_lr:
                self.logger.info('adjust lr at epoch {:d} from {:f} to {:f}'.format(self.iteration, current_lr, new_lr))

            for param_group in self.__optimizer.param_groups:
                param_group['lr'] = new_lr


    def __adjustWD(self):
        for param_group in self.__optimizer.param_groups:
            current_wd = param_group['weight_decay']
            break
        
        new_wd = current_wd

        if self.__conf.getHP('wd_mode') == 'linear':
            wd_max = self.__conf.getHP('wd_max')
            wd_min = self.__conf.getHP('wd_min')

            new_wd = wd_min + self.iteration * (wd_max - wd_min) / self.max_epoch

        for param_group in self.__optimizer.param_groups:
            param_group['weight_decay'] = new_wd
