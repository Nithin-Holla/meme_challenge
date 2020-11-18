import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import os
import json
from types import SimpleNamespace
from PIL import Image
from tqdm.notebook import tqdm
import time
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger()


class ObjectTextDataset(data.Dataset):
    
    def __init__(self, 
                 meme_filepath : str,
                 object_filepath : str,
                 object_to_text_filepath : str,
                 confidence_threshold = 0.5, # tuple or float
                 swap_prob : float = 0.0,
                 sep_token : str = "</s>",
                 join_token : str = ", ",
                 text_padding = None,
                 return_ids : bool = False,
                 debug : bool = False):
        super().__init__()
        self.meme_filepath = meme_filepath
        self.object_filepath = object_filepath
        self.object_to_text_filepath = object_to_text_filepath
        self.confidence_threshold = confidence_threshold
        self.swap_prob = swap_prob
        self.sep_token = sep_token
        self.join_token = join_token
        self.text_padding = text_padding
        self.return_ids = return_ids
        self.debug = debug
        
        if isinstance(self.confidence_threshold, tuple) and self.confidence_threshold[0] == self.confidence_threshold[1]:
            self.confidence_threshold = self.confidence_threshold[0]
        
        self._load_dataset()
    
    
    def _load_dataset(self):
        # Test filepaths
        assert os.path.isfile(self.meme_filepath), "Meme dataset file cannot be found: \"%s\"." % self.meme_filepath
        assert os.path.isfile(self.object_filepath), "Object dataset file cannot be found: \"%s\"." % self.object_filepath
        assert os.path.isfile(self.object_to_text_filepath), "Object to text file cannot be found: \"%s\"." % self.object_to_text_filepath
        assert self.meme_filepath.endswith(".jsonl"), "The meme filepath requires a JSON list file (\".jsonl\"). Please correct the given filepath \"%s\"" % self.meme_filepath
        assert self.object_filepath.endswith(".npz"), "The object filepath requires a npz file (\".npz\"). Please correct the given filepath \"%s\"" % self.object_filepath
        assert self.object_to_text_filepath.endswith(".json"), "The object to text filepath requires a JSON file (\".json\"). Please correct the given filepath \"%s\"" % self.object_to_text_filepath
        
        # Load json-list file
        with open(self.meme_filepath, "r") as f:
            json_list = f.readlines()
        json_list = [json.loads(json_str) for json_str in json_list]
        
        # Loading json files into namespace object
        # Note that if labels do not exist, they are replaced with -1
        self.data = SimpleNamespace(ids=None, imgs=None, labels=None, text=None)
        self.data.ids = torch.LongTensor([j["id"] for j in json_list])
        self.data.labels = torch.LongTensor([j.get("label", -1) for j in json_list])
        self.data.text = [j["text"] for j in json_list]
        
        assert self.data.ids.shape[0] == self.data.labels.shape[0] and \
               self.data.ids.shape[0] == len(self.data.text), "Internal error in loading. Data lists not equal length."
        
        ## Load object dataset
        arr = np.load(self.object_filepath)
        arr_ids, arr_objects, arr_probs = arr["ids"], arr["objects"], arr["probs"]
        arr_idx = np.zeros(self.data.ids.shape[0], dtype=np.int32)
        for i, data_id in enumerate(self.data.ids):
            idx_list = np.where(arr_ids == data_id.item())[0]
            assert len(idx_list) > 0, "Could not find ID in object file: %i." % data_id
            arr_idx[i] = idx_list[0]
        self.data.objects = arr_objects[arr_idx]
        self.data.object_probs = arr_probs[arr_idx]
        
        ## Load json dict converting object ids to text
        with open(self.object_to_text_filepath, "r") as f:
            self.object2text = json.load(f)
        self.object2text = {int(key): self.object2text[key] for key in self.object2text}
        
        
    def __len__(self):
        return len(self.data.ids)
    
    
    def __getitem__(self, idx):
        data_id = self.data.ids[idx]
        label = self.data.labels[idx]
        text = self.data.text[idx]
        
        object_text = self._create_object_text(idx)
        text = text + " %s " % self.sep_token + object_text
        
        if not self.return_ids:
            return text, label
        else:
            return text, label, data_id
        
        
    def _create_object_text(self, idx):
        # Step 1: Determine threshold
        if isinstance(self.confidence_threshold, tuple):
            thresh = np.random.uniform(low=self.confidence_threshold[0], high=self.confidence_threshold[1])
        else:
            thresh = self.confidence_threshold
        
        # Step 2: Objects to text
        objects = self.data.objects[idx, np.where(self.data.object_probs[idx] > thresh)[0]]
        objects = [self.object2text[objects[obj_idx]] for obj_idx in range(objects.shape[0])]
        
        # Step 3: Swap objects randomly
        if self.swap_prob > 0.0 and len(objects) > 1:
            order = np.random.permutation(len(objects)-1)
            for pos in order:
                if np.random.uniform() < self.swap_prob:
                    tmp = objects[pos]
                    objects[pos] = objects[pos+1]
                    objects[pos+1] = tmp
        
        # Step 4: Combine into a sentence
        object_text = self.join_token.join(objects)
        
        return object_text
    
    
    def get_collate_fn(self, num_imgs_to_check=20):
        """
        Returns functions to use in the Data loader (collate_fn).
        If images don't match to the same shape, we cannot stack them. Hence, we return a list instead.
        For text, the function "text_padding" takes all text elements, and is expected to return a list or stacked tensor.
        
        Inputs:
            num_imgs_to_check - Number of images we should check on the shape equality. Ideally, we should check all images, 
                                but this can take quite long if the images have not be preloaded. Hence, this input argument
                                defines the maximum to go (20 should be sufficient if we are not extremly unlucky)
        """
        
        def collate_fn(samples):
            texts = [s[0] for s in samples]
            if self.text_padding is not None:
                texts = self.text_padding(texts)
            
            other_params = tuple([torch.stack([s[i] for s in samples], dim=0) for i in range(1,len(samples[0]))])
            
            return (texts,) + other_params
        
        return collate_fn
    
    
    def get_by_id(self, data_id):
        """
        Function to access data point via their ID (given by the dataset) instead of range 0-N.
        """
        if data_id in self.data.ids:
            idx = np.where(self.data.ids == data_id)[0]
            return self.__getitem__(idx)
        else:
            logger.warning("Tried to access data id \"%s\", but is not present in the dataset." % str(data_id))
            return None
        
    
    def num_objects_over_threshold(self):
        threshold = np.arange(0, 1, 0.001)
        num_objects = (self.data.object_probs[None] > threshold[:,None,None]).sum(axis=2)
        mean_objects = num_objects.mean(axis=1)
        percentile_90 = np.percentile(num_objects, q=90, axis=-1)
        percentile_10 = np.percentile(num_objects, q=10, axis=-1)
        plt.plot(threshold, mean_objects, color="C0")
        # plt.fill_between(threshold, mean_objects, mean_objects*0.0, color="C0", alpha=0.5)
        
        plt.plot(threshold, percentile_90, color="C2")
        plt.plot(threshold, percentile_10, color="C2")
        plt.fill_between(threshold, percentile_10, percentile_90, color="C2", alpha=0.2)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,mean_objects.max()*1.1])
        plt.xlabel("Confidence threshold")
        plt.ylabel("Number of objects per image")
        plt.title("Number of objects per image over confidence threshold")
        plt.show()