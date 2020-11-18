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
from data.hatespeech_dataset import TwitterHatespeechDataset
from text_based.model import TransformerClassificationHead, MODEL_DICT
from utils.optim_utils import get_optimizer



class TrainerHatespeech(TrainerTemplate):


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
                                        num_classes=self.config['n_classes'])


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
        tokenized_text, self.batch_label = self.batch
        self.batch_label = self.batch_label.to(self.device)

        # Text input
        tokenized_text = {key: tokenized_text[key].to(self.device) for key in tokenized_text}
        self.preds = self.model(**tokenized_text)
        self.calculate_loss(self.preds, self.batch_label, grad_step=True)
        

    def eval_iter_step(self, iters, batch, test):
        tokenized_text, batch_label = batch
        batch_label = batch_label.to(self.device)

        # Forward pass
        tokenized_text = {key: tokenized_text[key].to(self.device) for key in tokenized_text}
        preds = self.model(**tokenized_text)

        self.calculate_loss(preds, batch_label, grad_step=False)


    def test_iter_step(self, batch):
        tokenized_text, batch_label = batch
        batch_label = batch_label.to(self.device)

        # Forward pass
        tokenized_text = {key: tokenized_text[key].to(self.device) for key in tokenized_text}
        preds = self.model(**tokenized_text)
        return preds.squeeze()





if __name__ == '__main__':
    defaults = {
        'lr': 5e-5,
        'warmup_steps': 200,
        'scheduler': 'warmup_cosine',
        'optimizer': 'adamw',
        'log_every': 50,
        'max_epoch': 50,
        'batch_size': 32,
        'loss_func': 'ce'
    }
    parser = argparse.ArgumentParser()
    TrainerTemplate.add_default_argparse(parser, defaults=defaults)
    
    #### Pre-processing Params ####
    parser.add_argument('--max_txt_len', type=int, default=256,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--model', type=str, default="BERT",
                        help='Name of the model to use (BERT, RoBERTa, ELECTRA, ALBERT, ...)')
    parser.add_argument('--lr_head', type=float, default=1e-4,
                        help='Learning rate for the MLP head')
    parser.add_argument('--num_layers_freeze', type=int, default=0,
                        help='Number of layers to freeze in BERT')

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        LOGGER.warning("There have been unprocessed parser arguments: " + str(unparsed))
    config = args.__dict__
    config = TrainerTemplate.preprocess_args(config)
    config['model'] = config['model'].lower()

    assert config['model'] in MODEL_DICT, "Given model is not known. Please choose between the following: " + str(MODEL_DICT.keys())
    config['model'] = MODEL_DICT[config['model']]
    # Tokenize
    tokenizer = config['model']["tokenizer"].from_pretrained(config['model']['pretrain'])
    tokenizer_func = partial(tokenizer, max_length=config['max_txt_len'], padding='longest',
                             truncation=True, return_tensors='pt')


    # Prepare the datasets and iterator for training
    train_dataset = TwitterHatespeechDataset(filepath=os.path.join(config['data_path'], 'train.csv'), 
                                             text_tokenize=tokenizer_func)
    test_dataset = TwitterHatespeechDataset(filepath=os.path.join(config['data_path'], 'test.csv'), 
                                            text_tokenize=tokenizer_func)
    config['n_classes'] = train_dataset.num_classes

    
    config['train_loader'] = data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=train_dataset.get_collate_fn(), shuffle=True, drop_last=True, pin_memory=True)
    config['val_loader'] = data.DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=test_dataset.get_collate_fn())
    config['test_loader'] = config['val_loader'] # No need of extra test set

    trainer = None
    try:
        trainer = TrainerHatespeech(config)
        trainer.train_main()
    except KeyboardInterrupt:
        LOGGER.warning("Keyboard interrupt by user detected at iteration %i...\nClosing the tensorboard writer!" % ((trainer.iters + trainer.total_iters) if trainer is not None else -1))
        config['writer'].close()
