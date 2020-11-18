import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data

import argparse
import os
import sys
sys.path.append("../")

from utils.logger import LOGGER
from train_template import TrainerTemplate
from data.meme_dataset import MemeDataset, ConfounderSampler
from text_based.model import TransformerClassificationHead, MODEL_DICT
from utils.optim_utils import get_optimizer
from utils.crossval import train_crossval



class TrainerLanguage(TrainerTemplate):


    def init_model(self):
        base_model = self.config['model']['class'].from_pretrained(self.config['model']['pretrain'])
        if self.config['num_layers_freeze'] > 0:
            for n, p in base_model.named_parameters():
                if n.startswith("encoder.layer"):
                    layer_num = int(n.split(".")[2]) # encoder.layer.##...
                    if layer_num < self.config['num_layers_freeze']:
                        print("Freezing %s..." % n)
                        p.requires_grad = False
        self.model = TransformerClassificationHead(
                                        num_layers=1, 
                                        hidden_dim=512, 
                                        act_fn=nn.GELU(), 
                                        base_model=base_model,
                                        use_pretrained_pool=False,
                                        dropout=0.5,
                                        num_classes=1)


    def load_model(self):
        # Load pretrained model
        if self.model_file:
            checkpoint = torch.load(self.model_file)
        else:
            checkpoint = {}
        self.model.load_state_dict(checkpoint['model_state_dict'])


    def init_optimizer(self):
        def group_param_func(named_params):
            base = {'params': [(n,p) for n,p in named_params if n.startswith("base_model")], "lr": config["lr"]}
            head = {'params': [(n,p) for n,p in named_params if not n.startswith("base_model")], "lr": config["lr_head"]}
            return [head, base]
        self.optimizer = get_optimizer(self.model, self.config, group_param_func=group_param_func)


    def train_iter_step(self):
        input_ids = self.batch["input_ids"]
        position_ids = self.batch["position_ids"]
        labels = self.batch["labels"]

        self.preds = self.model(input_ids=input_ids, position_ids=position_ids)
        self.calculate_loss(self.preds, labels, grad_step=True)
        

    def eval_iter_step(self, iters, batch, test):
        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        labels = batch["labels"]

        preds = self.model(input_ids=input_ids, position_ids=position_ids)
        self.calculate_loss(preds, labels, grad_step=False)


    def test_iter_step(self, batch):
        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        labels = batch["labels"]

        preds = self.model(input_ids=input_ids, position_ids=position_ids)
        return preds.squeeze()





if __name__ == '__main__':
    defaults = {
        'lr': 5e-5,
        'warmup_steps': 100,
        'scheduler': 'warmup_cosine',
        'optimizer': 'adamw',
        'log_every': 50,
        'max_epoch': 10,
        'batch_size': 32
    }
    parser = argparse.ArgumentParser()
    TrainerTemplate.add_default_argparse(parser, defaults=defaults)
    
    parser.add_argument('--max_txt_len', type=int, default=256,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--model', type=str, default="BERT",
                        help='Name of the model to use (BERT, RoBERTa, ELECTRA, ALBERT)')
    parser.add_argument('--lr_head', type=float, default=1e-4,
                        help='Learning rate for the MLP head')
    parser.add_argument('--num_layers_freeze', type=int, default=0,
                        help='Number of layers to freeze in BERT')


    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        LOGGER.warning("There have been unprocessed parser arguments: " + str(unparsed))
    config = args.__dict__
    config = TrainerTemplate.preprocess_args(config)
    # config['no_model_checkpoints'] = (config['no_model_checkpoints'] or config['debug'])
    config['model'] = config['model'].lower()

    assert config['model'] in MODEL_DICT, "Given model is not known. Please choose between the following: " + str(MODEL_DICT.keys())
    config['model'] = MODEL_DICT[config['model']]
    # Tokenize
    tokenizer = config['model']["tokenizer"].from_pretrained(config['model']['pretrain'])
    tokenizer_func = partial(tokenizer, max_length=config['max_txt_len'], padding='longest',
                             truncation=True, return_tensors='pt', return_length=True)


    # Prepare the datasets and iterator for training and evaluation
    def train_data_loader(train_file):
        if config['debug']:
            train_file = os.path.join(config["data_path"], "dev_seen.jsonl")
        train_dataset = MemeDataset(filepath=train_file, text_only=True, text_padding=tokenizer_func)
        return data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], 
                               collate_fn=train_dataset.get_collate_fn(), pin_memory=True, # shuffle is mutually exclusive with sampler. It is shuffled anyways
                               sampler=ConfounderSampler(train_dataset, repeat_factor=config["confounder_repeat"]))

    def val_data_loader(val_file):
        val_dataset = MemeDataset(filepath=val_file, text_only=True, text_padding=tokenizer_func)
        return data.DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=val_dataset.get_collate_fn())

    def test_data_loader(test_file):
        test_dataset = MemeDataset(filepath=test_file, text_only=True, text_padding=tokenizer_func, return_ids=True)
        return data.DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=test_dataset.get_collate_fn())
        
    config['test_loader'] = []
    for test_file in ['test_seen.jsonl', 'test_unseen.jsonl', 'dev_seen.jsonl', 'dev_unseen.jsonl']:
        config['test_loader'].append(test_data_loader(os.path.join(config['data_path'], test_file)))

    train_crossval(trainer_class=TrainerLanguage,
                   config=config,
                   data_loader_funcs={"train": train_data_loader, "val": val_data_loader, "test": test_data_loader},
                   num_folds=config['num_folds'],
                   dev_size=config['crossval_dev_size'],
                   use_dev_set=config['crossval_use_dev'])