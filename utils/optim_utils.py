import math
import sys
import torch
from torch.optim import Optimizer
from torch.optim import SGD, Adam, Adamax, AdamW



def get_optimizer(model, config, group_param_func=None):
    param_optimizer = list(model.named_parameters())
    if group_param_func is not None:
        optimizer_grouped_parameters = group_param_func(param_optimizer)
    else:
        optimizer_grouped_parameters = [{'params': param_optimizer}]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_grouped_parameters = []
    for g in optimizer_grouped_parameters:
        decay_params = {'params': [p for n, p in g['params'] if not any(nd in n for nd in no_decay)],
                        'weight_decay': config['weight_decay']}
        no_decay_params = {'params': [p for n, p in g['params'] if any(nd in n for nd in no_decay)],
                           'weight_decay': 0.0}
        for key in g:
            if not (key in ['params', 'weight_decay']):
                decay_params[key] = g[key]
                no_decay_params[key] = g[key] 

        new_grouped_parameters += [decay_params, no_decay_params]

    optimizer_grouped_parameters = new_grouped_parameters

    OptimKwargs = dict()
    if config['optimizer'] == 'adam':
        OptimCls = Adam
        OptimKwargs['betas'] = (config['beta1'], config['beta2'])
    elif config['optimizer'] == 'adamax':
        OptimCls = Adamax
    elif config['optimizer'] == 'adamw':
        OptimCls = AdamW
        OptimKwargs['betas'] = (config['beta1'], config['beta2'])
    elif config['optimizer'] == 'sgd':
        OptimCls = SGD
        OptimKwargs['momentum'] = config['beta1']
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters, lr=config['lr'], **OptimKwargs)
    return optimizer