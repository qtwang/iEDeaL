# coding = utf-8

import gc

import torch
import numpy as np

from util.conf import Conf
from util.train import EEGTrainable


class Experiment:
    def __init__(self, conf):
        if type(conf) is str:
            conf = Conf(conf)

        assert type(conf) is Conf
        self.__conf = conf

        self.__debug = self.__conf.getHP('debug')
        
        self.early_stop_tracebacks = self.__conf.getHP('early_stop_tracebacks')

        self.earlystop_type = self.__conf.getHP('earlystop_type')
        self.earlystop_target = self.__conf.getHP('earlystop_target')
        self.earlystop_threshold_vfbeta = self.__conf.getHP('earlystop_threshold_vfbeta')

        self.trainer = EEGTrainable(self.__conf, train=not self.__conf.query_only)


    def run(self, extra_epoch: int = -1) -> float:
        if not self.__conf.query_only:
            if extra_epoch == -1:
                max_epoch = self.__conf.getHP('num_epoch')
            else:
                max_epoch = self.trainer.iteration + extra_epoch

            earlystop_epsilon = self.__conf.getHP('earlystop_epsilon')
            earlystop_maxpatience = self.__conf.getHP('earlystop_patience')
            earlystop_patience = earlystop_maxpatience
            earlystop_patience_failure = earlystop_maxpatience

            clean, checkpoint = False, False
            tlosses, tfbetas, vfbetas = [], [], []
            
            while self.trainer.iteration < max_epoch:
                tloss, tfbeta, _, _, vfbeta, _, _ = self.trainer.step()
                
                if type(tloss) is tuple:
                    tloss = tloss[0]
                    
                if np.isnan(tloss):
                    clean, checkpoint = True, False
                    self.trainer.logger.info('training failed (nan)')
                    break

                if tfbeta < earlystop_epsilon:
                    earlystop_patience_failure -= 1
                    
                    if earlystop_patience_failure <= 0:
                        clean, checkpoint = True, False
                        self.trainer.logger.info('training failed (tf=0)')
                        break
                else:
                    earlystop_patience_failure = earlystop_maxpatience

                tlosses.append(tloss), tfbetas.append(tfbeta), vfbetas.append(vfbeta)
                
                if self.earlystop_type != 'none' and self.trainer.ready4stop() and len(vfbetas) > 1 and tfbeta >= vfbeta > self.earlystop_threshold_vfbeta:
                    if self.earlystop_type == 'continuous':
                        if self.earlystop_target == 'tloss':
                            if tlosses[-1] > tlosses[-2]:
                                earlystop_patience -= 1
                            else:
                                earlystop_patience = earlystop_maxpatience
                        elif self.earlystop_target == 'tfbeta':
                            if tfbetas[-2] > tfbetas[-1]:
                                earlystop_patience -= 1
                            else:
                                earlystop_patience = earlystop_maxpatience
                        elif self.earlystop_target == 'vfbeta':
                            if vfbetas[-2] > vfbetas[-1]:
                                earlystop_patience -= 1
                            else:
                                earlystop_patience = earlystop_maxpatience
                        else:
                            raise ValueError('illegal earlystop_target: {:s}'.format(self.earlystop_target))

                        if earlystop_patience <= 0:
                            clean, checkpoint = True, True
                            self.trainer.logger.info('training early stopped')
                            break
                    elif self.earlystop_type == 'mean':
                        if self.earlystop_target == 'tloss':
                            to_earlystop = len(tlosses) >= self.early_stop_tracebacks + 1 and tloss + earlystop_epsilon > np.mean(tlosses[-1 - self.early_stop_tracebacks: -1])
                        elif self.earlystop_target == 'tfbeta':
                            to_earlystop = len(tfbetas) >= self.early_stop_tracebacks + 1 and np.mean(tfbetas[-1 - self.early_stop_tracebacks: -1]) > tfbeta - earlystop_epsilon
                        elif self.earlystop_target == 'vfbeta':
                            to_earlystop = len(vfbetas) >= self.early_stop_tracebacks + 1 and np.mean(vfbetas[-1 - self.early_stop_tracebacks: -1]) > vfbeta - earlystop_epsilon 
                        else:
                            raise ValueError('illegal earlystop_target: {:s}'.format(self.earlystop_target))

                        if to_earlystop:
                            clean, checkpoint = True, True
                            self.trainer.logger.info('training early stopped')
                            break
                    else:
                        raise ValueError('illegal earlystop_type: {:s}'.format(self.earlystop_type))
                
            if self.trainer.iteration == max_epoch:
                clean, checkpoint = True, True
                self.trainer.logger.info('training finished')

            if self.__debug:
                if checkpoint:
                    self.trainer.log_predictions()

            if clean:
                self.trainer.cleanup(checkpoint)

                del self.trainer
                del self.__conf
                
                gc.collect()
                torch.cuda.empty_cache()
            
            if len(vfbetas) > 0:
                return vfbetas[-1]
            else:
                return vfbeta
