import argparse
import os

import pandas as pd
import shutil


def load_results(file):
    results_df = pd.read_csv(file)
    return results_df


def copy_misclassified_imgs(results_df, img_dir, save_dir):
    id_list = results_df.index[results_df['label'] != results_df['gt']].tolist()
    for idx in id_list:
        img_name = str(results_df['id'][idx]).zfill(5)
        img_file = os.path.join(img_dir, img_name + '.png')
        label = 'hateful' if results_df['gt'][idx] == 1 else 'not_hateful'
        assert os.path.isfile(img_file), 'Could not find image {}'.format(img_file)
        dir = os.path.join(save_dir, label)
        new_img_file = os.path.join(dir, img_name + '.png')
        shutil.copy(img_file, new_img_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, help='CSV file containing predictions', required=True)
    parser.add_argument('--img_dir', type=str, help='Directory containing original images')
    parser.add_argument('--save_dir', type=str, help='Directory to save misclassified images')

    args = parser.parse_args()

    # Load results file
    results_df = load_results(args.results_file)

    # Print misclassifications
    misclassified_ids = results_df['id'][results_df['label'] != results_df['gt']].values
    print('The following {} image IDs are misclassified: '.format(len(misclassified_ids)))
    print(misclassified_ids)

    # Save missclassified images to disk
    if args.save_dir is not None:
        assert args.img_dir, "Please specify the image directory"
        assert os.path.isdir(args.img_dir), "Invalid image directory"
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'hateful'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'not_hateful'), exist_ok=True)
        copy_misclassified_imgs(results_df, args.img_dir, args.save_dir)

