import os, torch, random, re, sys, time
import numpy as np
import random
import pandas as pd
from typing import List, Tuple

from torch.nn.utils.rnn import pad_sequence

from utils.logger import LOGGER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    return device

def calc_elapsed_time(start, end):
    hours, rem = divmod(end-start, 3600)
    time_hours, time_rem = divmod(end, 3600)
    minutes, seconds = divmod(rem, 60)
    time_mins, _ = divmod(time_rem, 60)
    return int(hours), int(minutes), seconds



def log_tensorboard(config, writer, model, epoch, iters, total_iters, loss, metrics=None, lr=0, thresh=0, loss_only=True, val=False):   
    model_log = model.module if config['parallel_computing'] == True else model
        
    if loss_only:
        writer.add_scalar('Train/Loss', sum(loss)/len(loss), ((iters+1)+ total_iters))
        # if iters%500 == 0:
        #     for name, param in model_log.encoder.named_parameters():
        #         print("\nparam {} grad = {}".format(name, param.grad.data.view(-1)))
        #         sys.exit()
        #         if not param.requires_grad or param.grad is None:
        #             continue
        #         writer.add_histogram('Iters/'+name, param.data.view(-1), global_step= ((iters+1)+total_iters))
        #         writer.add_histogram('Grads/'+ name, param.grad.data.view(-1), global_step = ((iters+1)+ total_iters))
    else:
        if not val:
            writer.add_scalar('Train/Epoch_Loss', sum(loss)/len(loss), ((iters+1)+ total_iters))
            writer.add_scalar('Train/F1', metrics['F1'], epoch)
            writer.add_scalar('Train/Precision', metrics['precision'], epoch)
            writer.add_scalar('Train/Recall', metrics['recall'], epoch)
            writer.add_scalar('Train/Accuracy', metrics['accuracy'], epoch)
            writer.add_scalar('Train/AUC-ROC', metrics['aucroc'], epoch)
            writer.add_scalar("Train/learning_rate", lr, epoch)
            
            # for name, param in model_log.encoder.named_parameters():
            #     if not param.requires_grad:
            #         continue
            #     writer.add_histogram('Epochs/' + name, param.data.view(-1), global_step= epoch)
        
            
        else:
            writer.add_scalar('Validation/Loss', loss, epoch)
            writer.add_scalar('Validation/F1', metrics['F1'], epoch)
            writer.add_scalar('Validation/Recall', metrics['recall'], epoch)
            writer.add_scalar('Validation/Precision', metrics['precision'], epoch)
            writer.add_scalar('Validation/Accuracy', metrics['accuracy'], epoch)
            writer.add_scalar('Validation/AUC-ROC', metrics['aucroc'], epoch)



def print_stats(config, epoch, train_metrics, train_loss, val_metrics, val_loss, start, lr):
    end = time.time()
    hours, minutes, seconds = calc_elapsed_time(start, end)
    
    train_loss = sum(train_loss)/len(train_loss)
    print("\nEpoch: {}/{},  \
          \ntrain_loss = {:.4f},  train_acc = {:.4f},  train_prec = {:.4f},  train_recall = {:.4f},  train_f1 = {:.4f},  train_aucroc = {:.4f},  train_opt_accuracy = {:.4f},  train_threshold = {:.4f}  \
          \neval_loss = {:.4f},  eval_acc = {:.4f},  eval_prec = {:.4f},  eval_recall = {:.4f},  eval_f1 = {:.4f},  eval_aucroc = {:.4f},  eval_opt_accuracy = {:.4f},  eval_threshold = {:.4f}  \
              \nlr  =  {:.8f}\nElapsed Time:  {:0>2}:{:0>2}:{:05.2f}"
                     .format(epoch, config['max_epoch'], train_loss, train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['F1'], train_metrics['aucroc'], train_metrics['optimal_accuracy'], train_metrics['optimal_threshold'],
                     val_loss, val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['F1'], val_metrics['aucroc'], val_metrics['optimal_accuracy'], val_metrics['optimal_threshold'], lr, hours,minutes,seconds))
        
        


def print_test_stats(test_metrics, val_metrics=None, test=False):
    def _print_res(metric_dict, val=False):
        if val:
            print("\n" + "-"*50 + "\nBest Validation scores:\n" + "-"*50)
            name = "Val"
        else:
            name = "Test"
        print("\n{} accuracy of best model = {:.3f}".format(name, metric_dict['accuracy']*100))
        print("{} AUC-ROC of best model = {:.3f}".format(name, metric_dict['aucroc']*100))
        print("{} precision of best model = {:.3f}".format(name, metric_dict['precision']*100))
        print("{} recall of best model = {:.3f}".format(name, metric_dict['recall']*100))
        print("{} f1 of best model = {:.3f}\n".format(name, metric_dict['F1']*100))

    if val_metrics is not None:
        _print_res(test_metrics, val=False)
        _print_res(val_metrics, val=True)
    else:
        _print_res(test_metrics, val=not test)
    


def set_seed(seed):
    # Seeds for reproduceable runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl + nbb] = torch.arange(max_len, max_len + nbb, dtype=torch.long).data
    return gather_index


def get_attention_mask(text_len, img_len):
    attn_mask = []
    for i in range(len(text_len)):
        attn_mask.append(torch.ones(text_len[i]+img_len[i]))
    attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
    return attn_mask
    

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def mean(l):
      return sum(l)/len(l) if len(l) > 0 else 0
