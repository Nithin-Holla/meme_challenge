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



class MetaLoader(object):
    """ wraps multiple data loaders """
    def __init__(self, loaders, accum_steps=1):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, data.DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n]*r)
        self.accum_steps = accum_steps
        self.step = 0

    def __iter__(self):
        """ this iterator will run indefinitely """
        task = self.sampling_pools[0]
        while True:
            if self.step % self.accum_steps == 0:
                task = random.choice(self.sampling_pools)
            self.step += 1
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_
            yield task, batch

    def __len__(self):
        return sum([d.__len__() for d in self.name2loader.values()])




class Pretrain_MemeDataset(MemeDataset):

    def _prepare_data_list(self):
        # Test filepath
        self.filepath_train = os.path.join(self.filepath, 'train.jsonl')
        self.filepath_dev = os.path.join(self.filepath, 'dev_seen.jsonl')
        self.filepath_memotion = os.path.join(self.filepath, 'memotion_dataset')
        self.filepath_memotion_all = os.path.join(self.filepath_memotion, 'all.jsonl')
        self.basepath = self.filepath

        data_paths = [self.filepath_train, self.filepath_dev]
        if self.use_memotion:
            data_paths.append(self.filepath_memotion_all)
        
        for path in data_paths:
            assert os.path.isfile(path), "Dataset file cannot be found: \"%s\"." % path
            assert path.endswith(".jsonl"), "The filepath requires a JSON list file (\".jsonl\"). Please correct the given filepath \"%s\"" % path

        # Load json-list file
        self.json_list = []
        lens = 0        
        for path in data_paths:
            with open(path, "r") as f:
                json_list_file = f.readlines()
                lens+=len(json_list_file)
            for json_str in json_list_file: 
                self.json_list.append(json.loads(json_str))
        
        assert len(self.json_list) == lens, "Size of the combined train and dev_seen dataset does not match the sum of their individual sizes"
        self._load_dataset()


