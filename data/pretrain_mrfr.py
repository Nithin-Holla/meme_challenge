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
from toolz.sandbox import unzip
import time
import logging
import matplotlib.pyplot as plt
from data.meme_dataset import MemeDataset
from utils.utils import get_attention_mask, get_device, pad_tensors, get_gather_index
from data.pretrain_meme_dataset import Pretrain_MemeDataset



class MRFR_MemeDataset(Pretrain_MemeDataset):

    def __init__(self, use_memotion, mask_prob, **kwargs):
        self.mask_prob = mask_prob
        self.use_memotion = use_memotion
        super().__init__(**kwargs)

    def _get_img_mask(self, mask_prob, num_bb):
        img_mask = [random.random() < mask_prob for _ in range(num_bb)]
        if not any(img_mask):
            # at least mask 1
            img_mask[random.choice(range(num_bb))] = True
        img_mask = torch.tensor(img_mask)
        return img_mask

    def _get_img_tgt_mask(self, img_mask, txt_len):
        z = torch.zeros(txt_len, dtype=torch.uint8)
        img_mask_tgt = torch.cat([z, img_mask], dim=0)
        return img_mask_tgt

    def _get_feat_target(self, img_feat, img_masks):
        img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)  # (n, m, d)
        feat_dim = img_feat.size(-1)
        feat_targets = img_feat[img_masks_ext].contiguous().view(-1, feat_dim)  # (s, d)
        return feat_targets

    def _mask_img_feat(self, img_feat, img_masks):
        img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
        img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
        return img_feat_masked

    def __getitem__(self, idx):
        img_feat, img_pos_feat, text, _ = super().__getitem__(idx)
        num_bb = img_feat.size(0)
        img_mask = self._get_img_mask(self.mask_prob, num_bb)

        return {
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'text': text,
            'img_mask': img_mask
        }

    def get_collate_fn(self):
        """
        Returns functions to use in the Data loader (collate_fn).
        Image features and position features are stacked (with padding) and returned.
        For text, the function "text_padding" takes all text elements, and is expected to return a list or stacked tensor.
        """

        def collate_fn(samples):
            """
            Return:
            :input_ids    (n, max_L) padded with 0
            :position_ids (n, max_L) padded with 0
            :img_feat     (n, max_num_bb, feat_dim)
            :img_pos_feat (n, max_num_bb, 7)
            :gather_index  (n, max_L+max_num_bb, 768)
            :attn_masks   (n, max_{L + num_bb}) padded with 0
            :img_masks    (n, max_num_bb) padded with 0
            :img_mask_tgt (n, max_{L + num_bb}) padded with 0
            :feat_targets (n, max_num_bb)
            """

            # Image features
            img_feats = [s['img_feat'] for s in samples]
            img_pos_feats = [s['img_pos_feat'] for s in samples]
            num_bbs = [f.size(0) for f in img_feats]
            img_feats = pad_sequence(img_feats, batch_first=True, padding_value=0)
            img_pos_feats = pad_sequence(img_pos_feats, batch_first=True, padding_value=0)

            # Text
            texts = [s['text'] for s in samples]
            tokenized_text = self.text_padding(texts)
            input_ids = tokenized_text['input_ids']
            text_len = tokenized_text['length'].tolist()
            position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

            # Image mask
            img_mask_tgts = []
            img_masks = [s['img_mask'] for s in samples]

            for i, tl in enumerate(text_len):
                img_mask_tgts.append(self._get_img_tgt_mask(img_masks[i], tl))

            img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
            img_mask_tgts = pad_sequence(img_mask_tgts, batch_first=True, padding_value=0)
            feat_targets = self._get_feat_target(img_feats, img_masks)
            img_feats = self._mask_img_feat(img_feats, img_masks)

            # Attention mask
            attn_masks = get_attention_mask(text_len, num_bbs)

            # Gather index
            out_size = attn_masks.shape[1]
            batch_size = attn_masks.shape[0]
            max_text_len = input_ids.shape[1]
            gather_index = get_gather_index(text_len, num_bbs, batch_size, max_text_len, out_size)

            batch = {'input_ids': input_ids,
                     'position_ids': position_ids,
                     'img_feat': img_feats,
                     'img_pos_feat': img_pos_feats,
                     'attn_masks': attn_masks,
                     'gather_index': gather_index,
                     'feat_targets': feat_targets,
                     'img_masks': img_masks,
                     'img_mask_tgt': img_mask_tgts}
            return batch

        return collate_fn