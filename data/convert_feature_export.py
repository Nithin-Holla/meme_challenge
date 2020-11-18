import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import os
import sys

def parse_numpy_file(input_file, output_dir):
    arr = np.load(input_file, allow_pickle=True)
    info_dict = arr['info'].item()
    info_dict['image_height'] = info_dict['image_h']
    info_dict['image_width'] = info_dict['image_w']
    info_dict['bbox'] =  arr['bbox']
    info_dict['objects'] = info_dict['objects_id']
    base_output_name = os.path.join(output_dir, input_file.split("/")[-1].rsplit(".",1)[0])
    np.save(base_output_name + "_info.npy", info_dict)
    np.save(base_output_name + ".npy", arr["x"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='Directory in which the feature files extraced by the FasterRCNN are stored.')
    parser.add_argument('--output_dir', type=str, 
                        help='Directory where to extract the new features to.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    input_files = sorted(glob(os.path.join(args.input_dir, "*.npz")))
    for f in tqdm(input_files):
        parse_numpy_file(f, args.output_dir)