import os
import re
import csv
import sys
import json
import argparse
import random
from random import shuffle
import logging
from glob import glob
import numpy as np
from statistics import mean
OFFSET_IDX = 1e5  # start roughly after memesdataset files max idx

logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger('CrossValLog')


def generate_jsonl_file(data_path, dev_size=300):
    random.seed(42)
    data_list = []
    read_dir = os.path.join(data_path,'labels.csv')
    img_feat_dir = os.path.join(data_path, 'img_feats')
    with open(read_dir, 'r', encoding='utf8') as read_file:
        rows = csv.DictReader(read_file)
        for row in rows:
            data_dict = {}
            id = int(row[''])+1+int(OFFSET_IDX)
            img_feat_path = os.path.join(img_feat_dir, str(id)+'.npy')
            img_feat_info_path = os.path.join(img_feat_dir, str(id)+'_info.npy')
            # Only if the img_feats exist we add it to the dataset
            if os.path.isfile(img_feat_path) and os.path.isfile(img_feat_info_path):
                data_dict['id'] = str(id)
                data_dict['img'] = 'images\/'+str(row['image_name'].replace('image_', ''))
                data_dict['label'] = 0
                text = row['text_corrected']
                text = text.replace('\n', ' ')
                text = re.sub(r"\b(?:https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+(?:[/?].*)?", "", text)  # removes most urls
                text = re.sub(r"(w{3}\.)*[a-zA-Z0-9]+\.{1}(co){1}[m]{0,1}\s{0,1}", "", text) # removes any.com urls
                text = re.sub(r"(w{3}\.)*[a-zA-Z0-9]+\.{1}(net){1}\s{0,1}", "", text) # removes any.net urls
                data_dict['text'] = text
                data_list.append(data_dict)
    
    logger.info("Total data points = {}".format(len(data_list)))
    write_dir = os.path.join(data_path, 'all.jsonl')
    logger.info("Writing the file at : {}".format(write_dir))
    export_jsonl(write_dir, data_list)



def export_jsonl(filepath, dict_list):
    s = "\n".join([json.dumps(d) for d in dict_list])
    with open(filepath, "w") as f:
        f.write(s)


def rename_img_feats(dir='../dataset/memotion_dataset/img_feats'):
    logger.info("Renaming img_feat files..")
    for root, dirs, files in os.walk(dir):
        for count, file in enumerate(files):
            src_file_path = os.path.join(root, file)
            id = re.findall("\d+", file)[0] # find the number in string (image_xxx.npy)
            renamed_file = str(int(id)+int(OFFSET_IDX))+'_info.npy' if 'info' in file else str(int(id)+int(OFFSET_IDX))+'.npy'
            contents = np.load(src_file_path, allow_pickle=True)
            dest_file_path = os.path.join(root, renamed_file)
            np.save(dest_file_path, contents, allow_pickle=True)
    logger.info("Renaming and saving done..")






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/memotion_dataset',
                        help='Path to folder of the meme dataset')

    args, unparsed = parser.parse_known_args()
    config = args.__dict__

    assert os.path.exists(config['data_path']), "[!] The provided data path does not exist!"
    generate_jsonl_file(data_path=config['data_path'])
    rename_img_feats()