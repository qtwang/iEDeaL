# coding = utf-8

# credit: ICLR16 All you need is a good init
# credit: https://github.com/ducha-aiki/LSUV-pytorch/tree/0c45eb2a9bd8978f13572c328f2f0d5d11939c99

from __future__ import print_function

import numpy as np
import torch
import torch.nn.init
import torch.nn as nn


gg = {}
gg['hook_position'] = 0
gg['total_fc_conv_layers'] = 0
gg['done_counter'] = -1
gg['hook'] = None
gg['act_dict'] = {}
gg['counter_to_apply_correction'] = 0
gg['correction_needed'] = False
gg['current_coef'] = 1.0


# Orthonorm init code is taked from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def svd_orthonormal(w, logger = None):
    shape = w.shape
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)#w;
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v

    if logger is not None:
        logger.info('LSUV: {:s}, {:s}'.format(str(shape), str(flat_shape)))
    else:
        print (shape, flat_shape)

    q = q.reshape(shape)
    return q.astype(np.float32)


def store_activations(self, input, output):
    gg['act_dict'] = output.data.cpu().numpy();
    #print('act shape = ', gg['act_dict'].shape)
    return


def add_current_hook(m):
    if gg['hook'] is not None:
        return
    if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.Linear)):
        #print 'trying to hook to', m, gg['hook_position'], gg['done_counter']
        if gg['hook_position'] > gg['done_counter']:
            gg['hook'] = m.register_forward_hook(store_activations)
            #print ' hooking layer = ', gg['hook_position'], m
        else:
            #print m, 'already done, skipping'
            gg['hook_position'] += 1
    return


def count_conv_fc_layers(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or (isinstance(m, nn.Linear)):
        gg['total_fc_conv_layers'] +=1
    return


def remove_hooks(hooks):
    for h in hooks:
        h.remove()
    return


def orthogonal_weights_init(m, logger = None):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if hasattr(m, 'weight'):
            w_ortho = svd_orthonormal(m.weight.data.cpu().numpy(), logger)
            m.weight.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass
        else:
            #nn.init.orthogonal(m.weight)
            w_ortho = svd_orthonormal(m.weight.data.cpu().numpy(), logger)
            #print w_ortho 
            #m.weight.data.copy_(torch.from_numpy(w_ortho))
            m.weight.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass
    return


def apply_weights_correction(m):
    if gg['hook'] is None:
        return
    if not gg['correction_needed']:
        return
    if (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Conv1d) or (isinstance(m, nn.Linear)):
        if gg['counter_to_apply_correction'] < gg['hook_position']:
            gg['counter_to_apply_correction'] += 1
        else:
            if hasattr(m, 'weight'):
                m.weight.data *= float(gg['current_coef'])
                gg['correction_needed'] = False
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data += float(gg['current_bias'])
            return
    return


def LSUVinit(model, data, needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = True, 
             needed_mean = 0., cuda = False, verbose = True, logger = None):
    logger = logger

    cuda = data.is_cuda

    gg['total_fc_conv_layers']=0
    gg['done_counter']= 0
    gg['hook_position'] = 0
    gg['hook'] = None
    
    model.eval()
    
    if cuda:
        model = model.cuda()
        data = data.cuda()
    else:
        model = model.cpu()
        data = data.cpu() 

    # if verbose: print( 'Starting LSUV')

    model.apply(count_conv_fc_layers)
    
    if verbose: 
        if logger is not None:
            logger.info('LSUV: {:d} layers to process'.format(gg['total_fc_conv_layers']))
        else:
            print ('Total layers to process:', gg['total_fc_conv_layers'])
    
    with torch.no_grad():
        if do_orthonorm:
            # model.apply(orthogonal_weights_init)
            model.apply(lambda m: orthogonal_weights_init(m, logger=logger))

            if verbose: 
                if logger is not None:
                    logger.info('LSUV: orthonorm finishes')
                else:
                    print ('Orthonorm done')
        
        if cuda:
            model = model.cuda()

        for layer_idx in range(gg['total_fc_conv_layers']):
            # if verbose: print (layer_idx)

            model.apply(add_current_hook)
            out = model(data, refine=True, detach=False)
            
            current_std = gg['act_dict'].std()
            current_mean = gg['act_dict'].mean()
            
            if verbose: 
                if logger is not None:
                    logger.info('LSUV: std of layer {:d} = {:f}'.format(layer_idx, current_std))
                else:
                    print ('std at layer ', layer_idx, ' = ', current_std)
            
            attempts = 0
            
            while (np.abs(current_std - needed_std) > std_tol):
                gg['current_coef'] =  needed_std / (current_std  + 1e-8)
                gg['current_bias'] =  needed_mean - current_mean * gg['current_coef']
                gg['correction_needed'] = True

                model.apply(apply_weights_correction)

                if cuda:
                    model = model.cuda()
                out = model(data, refine=True, detach=False)

                current_std = gg['act_dict'].std()
                current_mean = gg['act_dict'].mean()

                if verbose: 
                    if logger is not None:
                        logger.info('LSUV: std of layer {:d} = {:f}, mean = {:f}'.format(layer_idx, current_std, current_mean))
                    else:
                        print ('std at layer ', layer_idx, ' = ', current_std, 'mean = ', current_mean)

                attempts += 1
                if attempts > max_attempts:
                    if verbose: 
                        if logger is not None:
                            logger.info('LSUV: layer {:d} cannot converge in {:d} iterations'.format(layer_idx, max_attempts))
                        else:
                            print ('Cannot converge in ', max_attempts, 'iterations')
                    
                    break

            if gg['hook'] is not None:
                gg['hook'].remove()

            gg['done_counter']+=1
            gg['counter_to_apply_correction'] = 0
            gg['hook_position'] = 0
            gg['hook']  = None

            # if verbose: 
            #     if logger is not None:
            #         logger.info('LSUV: layer {:d} finishes'.format(layer_idx))
            #     else:
            #         print ('finish at layer',layer_idx )

        if verbose: 
            if logger is not None:
                logger.info('LSUV: initialization finishes')
            else:
                print ('LSUV init done!')

        if not cuda:
            model = model.cpu()

    return model
