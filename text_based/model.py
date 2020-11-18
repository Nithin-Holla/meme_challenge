import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import ElectraModel, ElectraTokenizer

MODEL_DICT = {
    "bert": {"class": BertModel, "tokenizer": BertTokenizer, "pretrain": 'bert-base-uncased'},
    "bert_large": {"class": BertModel, "tokenizer": BertTokenizer, "pretrain": 'bert-large-uncased'},
    "roberta": {"class": RobertaModel, "tokenizer": RobertaTokenizer, "pretrain": 'roberta-base'},
    "roberta_large": {"class": RobertaModel, "tokenizer": RobertaTokenizer, "pretrain": 'roberta-large'},
    "roberta_mnli": {"class": RobertaModel, "tokenizer": RobertaTokenizer, "pretrain": 'roberta-large-mnli'},
    "albert": {"class": AlbertModel, "tokenizer": AlbertTokenizer, "pretrain": 'albert-xlarge-v2'},
    "albert_large": {"class": AlbertModel, "tokenizer": AlbertTokenizer, "pretrain": 'albert-xxlarge-v2'},
    "electra": {"class": ElectraModel, "tokenizer": ElectraTokenizer, "pretrain": 'google/electra-small-discriminator'}
}

class TransformerClassificationHead(nn.Module):

	def __init__(self, base_model, num_classes, 
					   num_layers=1, 
					   hidden_dim=512, 
					   act_fn=None, 
					   dropout=0.0,
					   use_pretrained_pool=False):
		super().__init__()
		self.base_model = base_model
		self.use_pretrained_pool = use_pretrained_pool
		input_dim = self.base_model.config.hidden_size
		act_fn = nn.ReLU(inplace=True) if act_fn is None else act_fn

		layers = [nn.Dropout(dropout)]
		for l in range(num_layers):
			layers += [nn.Linear(input_dim, hidden_dim), nn.Dropout(dropout), act_fn, nn.LayerNorm(hidden_dim)]
			input_dim = hidden_dim
		layers += [nn.Linear(input_dim, num_classes)]
		self.class_head = nn.Sequential(*layers)

	def forward(self, *args, **kwargs):
		out = self.base_model(*args, **kwargs)
		if isinstance(out, tuple) and len(out)>1:
			token_level_out, pool_out = out
			class_out = token_level_out[:,0] if self.use_pretrained_pool else pool_out
		else:
			class_out = out[0][:,0]
		preds = self.class_head(class_out)
		return preds