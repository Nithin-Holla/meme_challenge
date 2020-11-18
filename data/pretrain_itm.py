import torch
import torch.utils.data as data
import numpy as np
import os
import sys
import json
import random
from types import SimpleNamespace
from PIL import Image
from tqdm.notebook import tqdm
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import time
import logging
from transformers import BertTokenizer
from data.pretrain_meme_dataset import Pretrain_MemeDataset
from utils.utils import get_attention_mask, get_device, pad_tensors, get_gather_index


class ITM_MemeDataset(Pretrain_MemeDataset):
    def __init__(self, use_memotion, replace_prob, **kwargs):
        self.replace_prob = replace_prob
        self.use_memotion = use_memotion
        super().__init__(**kwargs)
        
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        label = 1

        # Replace text of an img_feat with prob=replace_prob
        replace = random.random()
        if replace < self.replace_prob:
            while True:
                rand_idx = random.choice([i for i in range(len(self.data.ids)) if i != idx])
                if item['text'] != self.data.text[rand_idx]:
                    break
            label = 0
            text = self.data.text[rand_idx]
        else:
            text = item['text']
            
        return {
            'img_feat' : item['img_feat'],
            'img_pos_feat' : item['img_pos_feat'],
            'text' : text,
            'label' : label}



    def get_collate_fn(self):
        def collate_fn(samples):
            img_feats = [s['img_feat'] for s in samples]
            img_pos_feats = [s['img_pos_feat'] for s in samples]
            texts = [s['text'] for s in samples]
            labels = [s['label'] for s in samples]

            texts = self.text_padding(texts)
            input_ids = texts['input_ids']
            text_len = texts['length'].tolist()
            img_len = [i.size(0) for i in img_feats]
            attn_masks = get_attention_mask(text_len, img_len)
            ot_inputs = torch.zeros(len(labels))
            labels = torch.tensor(labels)

            # text batches
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
                'targets': labels,
                'ot_inputs': ot_inputs}

            return batch        
        return collate_fn






if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="Path to jsonl file of dataset", required=True)
    parser.add_argument('--feature_dir', type=str, help='Directory containing image features', required=True)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer_func = partial(tokenizer, max_length=128, padding='max_length',
                             truncation=True, return_tensors='pt', return_length=True)

    dataset = ITM_MemeDataset(replace_prob=0.5,
                            filepath=args.filepath,
                            feature_dir=args.feature_dir,
                            preload_images=False, debug=True, text_padding=tokenizer_func)

    data_loader = data.DataLoader(dataset, batch_size=16, collate_fn=dataset.get_collate_fn())
    try:
        batch, label = next(iter(data_loader))
        logger.info("Data loading has been successful.")
    except Exception as e:
        logger.error("Error occured during data loading, please have a look at this:\n" + str(e))