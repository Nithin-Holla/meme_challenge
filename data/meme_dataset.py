import torch
import torch.utils.data as data
import os
import json
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence
from tqdm.notebook import tqdm
import time
import logging
from random import shuffle
try:
    from data.dataset_template import Dataset_Template
except ModuleNotFoundError as e:
    import sys
    sys.path.append("../")
    from data.dataset_template import Dataset_Template
from utils.utils import get_gather_index, get_attention_mask


logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger('MemeDatasetLog')



class MemeDataset(Dataset_Template):

    def _prepare_data_list(self):
        # Test filepath
        assert os.path.isfile(self.filepath), "Dataset file cannot be found: \"%s\"." % self.filepath
        assert self.filepath.endswith(".jsonl"), "The filepath requires a JSON list file (\".jsonl\"). Please correct the given filepath \"%s\"" % self.filepath
        self.basepath = self.filepath.rsplit("/",1)[0]
        # Load json-list file
        with open(self.filepath, "r") as f:
            self.json_list = f.readlines()
        self.json_list = [json.loads(json_str) for json_str in self.json_list]
        self._load_dataset()


    def _load_object2text(self):
        assert os.path.isfile(self.object_to_text_filepath), 'Cannot find file {}'.format(self.object_to_text_filepath)
        ## Load json dict converting object ids to text
        with open(self.object_to_text_filepath, "r") as f:
            self.object2text = json.load(f)
        self.object2text = {int(key): self.object2text[key] for key in self.object2text}


    def _load_dataset(self):        
        # Loading json files into namespace object
        # Note that if labels do not exist, they are replaced with -1        
        self.data = SimpleNamespace(ids=None, imgs=None, labels=None, text=None)
        self.data.ids = torch.LongTensor([int(j["id"]) for j in self.json_list])
        self.data.labels = torch.LongTensor([j.get("label", -1) for j in self.json_list])
        self.data.text = [j["text"] for j in self.json_list]
        self.data.imgs = [os.path.join(self.basepath, j["img"]) for j in self.json_list]
        self.img_feat = None
        self.img_pos_feat = None
        assert self.data.ids.shape[0] == self.data.labels.shape[0] and \
               len(self.data.text) == len(self.data.imgs) and \
               self.data.ids.shape[0] == len(self.data.imgs), "Internal error in loading. Data lists not equal length."
        
        # # Check that all images and features actually exist (prevent errors later in the training)
        # for i, img_filepath in enumerate(self.data.imgs):
        #     assert os.path.isfile(img_filepath), "Image filepath \"%s\" (%i element) does not exist. Make sure that all images exist in the dataset." % (img_filepath, i)
        if not self.text_only:
            for img_id in self.data.ids:
                img_id = self._expand_id(img_id.item())
                feature_file_name = os.path.join(self.feature_dir, '{}.npy'.format(img_id))
                feature_info_file = os.path.join(self.feature_dir, '{}_info.npy'.format(img_id))
                assert os.path.isfile(feature_file_name), "Feature file for image {} does not exist.".format(img_id)
                assert os.path.isfile(feature_info_file), "Feature info file for image {} does not exist.".format(img_id)

        # Load all images in RAM if selected
        if not self.text_only and self.preload_images:
            logger.info("Loading image features...")
            start_time = time.time()
            loaded_img_feats = []
            loaded_img_pos_feats = []
            loaded_objects = []
            loaded_objects_conf = []
            for img_id in tqdm(self.data.ids, desc="Loading images") if self.debug else self.data.ids:
                img_feat, img_pos_feat, objects, objects_conf = self._load_img_feature(img_id.item())
                loaded_img_feats.append(img_feat)
                loaded_img_pos_feats.append(img_pos_feat)
                loaded_objects.append(objects)
                loaded_objects_conf.append(objects_conf)
            self.img_feat = loaded_img_feats
            self.img_pos_feat = loaded_img_pos_feats
            self.objects = loaded_objects
            self.objects_conf = loaded_objects_conf
            logger.info("Finished loading %i image features in %i seconds." % (len(self.data.imgs), int(time.time() - start_time)))
        
        # Preprocess text if selected
        if self.text_preprocess is not None:
            self.data.text = self.text_preprocess(self.data.text)
        
    def __len__(self):
        return len(self.data.ids)


    def _get_object_text(self, objects):
        obj_text = [self.object2text[obj] for obj in objects]
        return ' '.join(obj_text)
    
    
    def __getitem__(self, idx):
        data_id = self.data.ids[idx]
        label = self.data.labels[idx]

        if self.text_only:
            img_feat, img_pos_feat = None, None
        else:
            if not self.preload_images:
                img_feat, img_pos_feat, objects, objects_conf = self._load_img_feature(data_id.item(), normalize=True)
            else:
                img_feat = self.img_feat[idx]
                img_pos_feat = self.img_pos_feat[idx]
                objects = self.objects[idx]
                objects_conf = self.objects_conf[idx]
            if self.confidence_threshold > 0.0:
                valid_boxes = (objects_conf > self.confidence_threshold)
                img_feat = img_feat[valid_boxes]
                img_pos_feat = img_pos_feat[valid_boxes]
                objects = objects[valid_boxes]
                objects_conf = objects_conf[valid_boxes] # Not really needed, but in case someone needs it
        
        text = self.data.text[idx]

        if self.include_object_tags:
            text = [text, self._get_object_text(objects)]

        if self.text_getitem is not None:
            text = self.text_getitem(text)

        return {
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'text': text,
            'label': label,
            'data_id': data_id}

    
    
    def get_collate_fn(self):
        """
        Returns functions to use in the Data loader (collate_fn).
        Image features and position features are stacked (with padding) and returned.
        For text, the function "text_padding" takes all text elements, and is expected to return a list or stacked tensor.
        """
        
        def collate_fn(samples):
            img_feat = [s['img_feat'] for s in samples]
            img_pos_feat = [s['img_pos_feat'] for s in samples]
            texts = [s['text'] for s in samples]
            labels = [s['label'] for s in samples]
            data_id = [s['data_id'] for s in samples]

            # Pad image feats
            if not self.text_only:
                img_feat = pad_sequence(img_feat, batch_first=True, padding_value=0)
                img_pos_feat = pad_sequence(img_pos_feat, batch_first=True, padding_value=0)
            else:
                img_feat = None
                img_pos_feat = None

            # Tokenize and pad text
            if self.text_padding is not None:
                texts = self.text_padding(texts)
            
            # Stack labels and data_ids
            labels = torch.stack(labels, dim=0)
            data_ids = torch.stack(data_id, dim=0)

            # Text input
            input_ids = texts['input_ids']
            text_len = texts['length'].tolist()
            token_type_ids = texts['token_type_ids'] if 'token_type_ids' in texts else None
            position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0).repeat(input_ids.shape[0], 1)

            # Attention mask
            if img_feat is None:
                attn_mask = texts['attention_mask']
                img_len = [0]
            elif self.compact_batch:
                img_len = [i.size(0) for i in img_feat]
                attn_mask = get_attention_mask(text_len, img_len)
            else:
                text_mask = texts['attention_mask']
                img_len = [i.size(0) for i in img_feat]
                zero_text_len = [0] * len(text_len)
                img_mask = get_attention_mask(zero_text_len, img_len)
                attn_mask = torch.cat((text_mask, img_mask), dim=1)

            # Gather index
            out_size = attn_mask.shape[1]
            batch_size = attn_mask.shape[0]
            max_text_len = input_ids.shape[1]
            if not self.text_only:
                gather_index = get_gather_index(text_len, img_len, batch_size, max_text_len, out_size)
            else:
                gather_index = None

            batch = {'input_ids': input_ids,
                    'position_ids': position_ids,
                    'img_feat': img_feat,
                    'img_pos_feat': img_pos_feat,
                    'token_type_ids': token_type_ids,
                    'attn_mask': attn_mask,
                    'gather_index': gather_index,
                    'labels': labels,
                    'ids' : data_ids}
            
            return batch        
        return collate_fn
    




class ConfounderSampler(data.Sampler):

    def __init__(self, dataset, repeat_factor : int = 1):
        super().__init__(dataset)
        logger.info("Setting up Confounder Sampler with repeat factor %i..." % repeat_factor)
        self.dataset = dataset
        self.repeat_factor = repeat_factor
        self._find_confounders()
        self._generate_sample_list()


    def _find_confounders(self):
        label_per_text = {}
        for idx, text in enumerate(self.dataset.data.text):
            if text not in label_per_text:
                label_per_text[text] = []
            label_per_text[text].append(self.dataset.data.labels[idx].item())
        confounder_text = [key for key in label_per_text if sorted(list(set(label_per_text[key]))) == [0,1]]
        self.non_confounders, self.confounders = [], []
        for idx, text in enumerate(self.dataset.data.text):
            if text in confounder_text:
                self.confounders.append(idx)
            else:
                self.non_confounders.append(idx)
            num_confounders, num_other = len(self.confounders), len(self.non_confounders)
        logger.info("Found %i text confounders and %i non-confounders in dataset %s (%i examples, %4.2f%% confounders)" % (num_confounders, num_other, self.dataset.name, len(self.dataset), 100.0*num_confounders/len(self.dataset)))


    def _generate_sample_list(self):
        plain_list = self.non_confounders[:]
        shuffle(plain_list)
        splits = [(len(plain_list)//self.repeat_factor)*i for i in range(self.repeat_factor)]
        splits.append(len(plain_list))

        sample_list = []
        for i in range(self.repeat_factor):
            sub_list = plain_list[splits[i]:splits[i+1]]
            sub_list += self.confounders
            shuffle(sub_list)
            sample_list += sub_list

        self.sample_list = sample_list


    def __iter__(self):
        self._generate_sample_list()
        return iter(self.sample_list)


    def __len__(self):
        return len(self.sample_list)




if __name__ == '__main__':
    import argparse
    from functools import partial
    from transformers import BertTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="Path to jsonl file of dataset", required=True)
    parser.add_argument('--feature_dir', type=str, help='Directory containing image features', required=True)
    args = parser.parse_args()

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer_func = partial(tokenizer, max_length=256, padding='max_length',
                             truncation=True, return_tensors='pt', return_length=True)
    dataset = MemeDataset(filepath=args.filepath,
                          feature_dir=args.feature_dir,
                          text_padding=tokenizer_func,
                          preload_images=False,
                          confidence_threshold=0.4,
                          debug=True)
    data_loader = data.DataLoader(dataset, batch_size=32, collate_fn=dataset.get_collate_fn(), sampler=ConfounderSampler(dataset, repeat_factor=2))
    logger.info("Length of data loader: %i" % len(data_loader))
    try:
        out_dict = next(iter(data_loader))
        logger.info("Data loading has been successful.")
    except NotImplementedError as e:
        logger.error("Error occured during data loading, please have a look at this:\n" + str(e))
    print("Image features", out_dict['img_feat'].shape)
