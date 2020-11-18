import torch
import torch.utils.data as data
import numpy as np
import json 
import csv
import re
from types import SimpleNamespace
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger('TwitterDatasetLog')

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+"
)

class TwitterHatespeechDataset(data.Dataset):
    
    def __init__(self, filepath : str,
                       text_tokenize = None):
        super().__init__()
        self.filepath = filepath
        self.text_tokenize = text_tokenize
        
        self._load_dataset()
        

    def _load_dataset(self):
        assert os.path.isfile(self.filepath), "Dataset file cannot be found: \"%s\"." % self.filepath
        assert self.filepath.endswith(".csv"), "Dataset file is expected to be a CSV file. Please correct the filepath \"%s\"." % self.filepath
        
        self.data = SimpleNamespace(labels=[], text=[])
        with open(self.filepath, 'r', newline='') as f:
            file_reader = csv.reader(f, delimiter=',')
            rows = list(file_reader)
        
        keys = rows[0]
        label_index = keys.index('label')
        text_index = keys.index('text')
        for row in rows[1:]:
            self.data.labels.append(row[label_index])
            self.data.text.append(row[text_index])
        
        self.data.text = [TwitterHatespeechDataset.preprocess_tweet(t) for t in self.data.text]
        self.label_names = sorted(list(set(self.data.labels)))
        self.data.labels = torch.LongTensor([self.label_names.index(l) for l in self.data.labels])
        self.num_classes = len(self.label_names)
            

    def __len__(self):
        return self.data.labels.shape[0]
    
    
    def __getitem__(self, idx):
        label = self.data.labels[idx]
        text = self.data.text[idx]
        
        return text, label
    
    
    def get_collate_fn(self):
        """
        Returns functions to use in the Data loader (collate_fn).
        For text, the function "text_tokenize" takes all text elements, and is expected to return a list or stacked tensor.
        """
        def collate_fn(samples):
            texts = [s[0] for s in samples]
            if self.text_tokenize is not None:
                texts = self.text_tokenize(texts)
            
            labels = torch.stack([s[1] for s in samples])
            return (texts, labels)
        
        return collate_fn
    

    @staticmethod
    def preprocess_tweet(tweet):
    	# Remove MKR. That hashtag is just too correlated with sexism
        tweet = tweet.replace("#MKR", "") 
        # Remove URLs and hashtags, as they are not expected in memes => don't have them during pre-training
        tweet = re.sub(r'https?://\S+', '', tweet) 
        tweet = re.sub(r'#[\w-]+', '', tweet)
        # Retweets
        tweet = re.sub(r'^["\']?RT @\S+:', '', tweet)
        tweet = re.sub(r'RT @\S+:', 'RT:', tweet)
        # User mentions (ignored as not relevant for non-twitter based data)
        tweet = re.sub(r'@\S+', '', tweet)
        # Remove emojis
        tweet = EMOJI_PATTERN.sub(r'', tweet)
        # Any additional, unnecessary characters
        tweet = tweet.replace("  "," ")
        tweet = tweet.replace("\\\'","'")
        tweet = tweet.strip('"\' \t\n')
        return tweet


