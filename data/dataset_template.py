import torch
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import logging
import matplotlib.pyplot as plt


logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger('MemeDatasetLog')



class Dataset_Template(data.Dataset):
    
    def __init__(self, 
                 filepath : str,
                 feature_dir: str = None,
                 preload_images : bool = True,
                 text_preprocess = None,
                 text_getitem = None,
                 text_padding = None,
                 text_only : bool = False,
                 return_ids : bool = False,
                 include_object_tags: bool = False,
                 object_to_text_file_path: str = '',
                 compact_batch: bool = True,
                 confidence_threshold : float = 0.0,
                 debug : bool = False):
        """
        Inputs:
            filepath - Filepath to the ".jsonl" file which stores a list of all data points.
            feature_dir - Directory containing image features.
            preload_images - If True, the dataset will load all images from the hard disk into RAM during creation of the object. 
                             Saves time during training, but takes longer to create the dataset object and requires more memory.
            text_preprocess - Function to execute after loading the text. Takes as input a list of all meme text in the dataset.
                              Is expected to return a list of processed text elements.
            text_getitem - Function to apply on each text element during "__getitem__" method of the dataset.
            text_padding - Function to apply for padding the input text to match a batch tensor.
            text_only - If True, only text will be loaded from the dataset. In the __getitem__, None values will be returned for images.
            return_ids - If True, the __getitem__ function returns the ids of the data points as last element
            include_object_tags - If True, include the object tags in the text
            object_to_text_filepath - File containing mapping between object IDs and object text
            compact_batch - If True, batches with text and image will be compacted without padding between them
            confidence_threshold - Threshold of object confidence under which bounding boxes are ignored
            debug - If True, more output information is logged. For instance, it displays a tqdm loading bar when preloading images.
        """
        super().__init__()
        self.filepath = filepath
        self.name = filepath.split("/")[-1].split(".")[0]
        self.feature_dir = feature_dir
        self.preload_images = preload_images
        self.text_preprocess = text_preprocess
        self.text_getitem = text_getitem
        self.text_padding = text_padding
        self.text_only = text_only
        self.include_object_tags = include_object_tags
        self.return_ids = return_ids
        self.compact_batch = compact_batch
        self.confidence_threshold = confidence_threshold
        self.debug = debug

        self._prepare_data_list()

        if self.include_object_tags:
            self.object_to_text_filepath = object_to_text_file_path
            self._load_object2text()
    
    
    def _prepare_data_list(self):
        raise NotImplementedError

    

    def _load_dataset(self):        
        raise NotImplementedError
    
    
    def _load_img(self, img_filepath):
        with open(img_filepath, "rb") as f:
            img = Image.open(f).convert("RGB")
        return img


    def _expand_id(self, img_id):
        return str(img_id).zfill(5)


    def _load_img_feature(self, img_id, normalize=False):
        img_id = self._expand_id(img_id)
        feature_file_name = os.path.join(self.feature_dir, '{}.npy'.format(img_id))
        feature_info_file = os.path.join(self.feature_dir, '{}_info.npy'.format(img_id))
        img_feat = torch.from_numpy(np.load(feature_file_name))
        img_feat_info = np.load(feature_info_file, allow_pickle=True).item()
        x1, y1, x2, y2 = np.split(img_feat_info['bbox'], 4, axis=1)
        img_width = img_feat_info['image_width']
        img_height = img_feat_info['image_height']
        objects = img_feat_info['objects']
        if 'objects_conf' in img_feat_info:
            objects_conf = img_feat_info['objects_conf']
        else:
            objects_conf = img_feat_info['cls_prob'].max(axis=-1)
        if normalize:
            x1 /= img_width
            x2 /= img_width
            y1 /= img_height
            y2 /= img_height
        w = x2 - x1
        h = y2 - y1
        img_pos_feat = torch.from_numpy(np.concatenate((x1, y1, x2, y2, w, h, w*h), axis=1))
        return img_feat, img_pos_feat, objects, objects_conf
        
        
    def __len__(self):
        raise NotImplementedError
    
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    
    def get_collate_fn(self):
        """
        Returns functions to use in the Data loader (collate_fn).
        Image features and position features are stacked (with padding) and returned.
        For text, the function "text_padding" takes all text elements, and is expected to return a list or stacked tensor.
        """
        
        def collate_fn(samples):
            raise NotImplementedError
        
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
        
        
    def show_img(self, idx=None, data_id=None, img=None):
        """
        Displays an image of the dataset. At least one of the following input arguments needs to be given:
            idx - Index of the image in the dataset
            data_id - ID of the data point with the image (stated in the original .jsonl file)
            img - PIL image to display
        """
        if img is None:
            if idx is None:
                if data_id is not None:
                    idx = np.where(self.data.ids == data_id)[0]
                else:
                    logger.warning("Method show_img got all empty inputs. No image displayed.")
                    return
            img = self.data.imgs[idx]
            label = self.data.labels[idx]
            if not self.preload_images:
                img = self._load_img(img)
        else:
            label = None
        
        plt.imshow(img)
        plt.axis('off')
        if label is not None:
            plt.title("Label: %s (%i)" % ("Hateful" if label==1 else ("Not hateful" if label==0 else "Unknown"), label))
        plt.show()
        plt.close()
