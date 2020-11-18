import torch
import torch.utils.data as data
import numpy as np
import os
import json
import random
from types import SimpleNamespace
from PIL import Image
from tqdm.notebook import tqdm
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
# from toolz.sandbox import unzip
import time
import logging
import matplotlib.pyplot as plt
from data.meme_dataset import MemeDataset
from utils.utils import get_attention_mask, get_device, pad_tensors, get_gather_index
from data.pretrain_meme_dataset import Pretrain_MemeDataset



class MLM_MemeDataset(Pretrain_MemeDataset):

    def __init__(self, use_memotion, mask_prob, cls_token, mask_token, sep_token, vocab_range, pad_token=0, **kwargs):
        self.mask_prob = mask_prob
        self.vocab_range = vocab_range
        self.mask_token = mask_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.use_memotion = use_memotion
        super().__init__(**kwargs)

    
    def get_masked_txt(self, tokens, vocab_range, mask):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :param vocab_range: for choosing a random word
        :return: (list of int, list of int), masked tokens and related labels for LM prediction
        """
        output_label = []
        for i, token in enumerate(tokens):
            token = token.item()
            if token in [self.cls_token, self.sep_token, self.pad_token]:
                output_label.append(-1)
                continue

            self.prob = random.random()
            # mask token with 15% probability
            if self.prob < self.mask_prob:
                self.prob /= self.mask_prob
                # 80% randomly change token to mask token
                if self.prob < 0.8:
                    tokens[i] = mask
                # 10% randomly change token to random token
                elif self.prob < 0.9:
                    tokens[i] = random.choice(list(range(*vocab_range)))
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
        if all(o == -1 for o in output_label):
            # at least mask 1 (the first word excluding the CLS token)
            output_label[1] = tokens[1]
            tokens[1] = mask
        return tokens, output_label


    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = self.get_masked_txt(input_ids, self.vocab_range, self.mask_token)
        return input_ids, txt_labels

    

    def get_collate_fn(self):
        def collate_fn(samples):
            """
            Return:
            :input_ids    (n, max_L) padded with 0
            :position_ids (n, max_L) padded with 0
            :img_feat     (n, max_num_bb, feat_dim)
            :img_pos_feat (n, max_num_bb, 7)
            :gather_index  (n, max_L+max_num_bb, 768)
            :attn_masks   (n, max_{L + num_bb}) padded with 0
            :txt_labels   (n, max_L) padded with -1
            """

            # (img_feats, img_pos_feats, text) = map(list, unzip(samples))
            img_feats = [inp['img_feat'] for inp in samples]
            img_pos_feats = [inp['img_pos_feat'] for inp in samples]
            text = [inp['text'] for inp in samples]

            text = self.text_padding(text)

            input_ids = text['input_ids']
            text_len = text['length'].tolist()
            txt_labels = []
            for i in range(input_ids.size(0)):
                input_ids[i], masked_txt_labels = self.create_mlm_io(input_ids[i])
                txt_labels.append(torch.tensor(masked_txt_labels))

            img_len = [i.size(0) for i in img_feats]
            attn_masks = get_attention_mask(text_len, img_len)

            # text batches
            txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
            position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

            # image batches
            num_bbs = [f.size(0) for f in img_feats]
            if not self.text_only:
                img_feat = pad_tensors(img_feats, num_bbs)
                img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
            attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

            bs, max_tl = input_ids.size()
            out_size = attn_masks.size(1)
            gather_index = get_gather_index(text_len, num_bbs, bs, max_tl, out_size)

            batch = {'input_ids': input_ids,
                    'position_ids': position_ids,
                    'img_feat': img_feat,
                    'img_pos_feat': img_pos_feat,
                    'attn_masks': attn_masks,
                    'gather_index': gather_index,
                    'txt_labels': txt_labels}
            return batch
        
        return collate_fn