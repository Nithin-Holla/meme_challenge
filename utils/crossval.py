import os
import json
import argparse
import random
import numpy as np
import math
from random import shuffle
import logging
from glob import glob
from statistics import mean
from collections import defaultdict
try:
    from utils.utils import set_seed
    from utils.ensemble import find_ensemble
except ModuleNotFoundError:
    print("Utility modules were not found.")

logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger('CrossValLog')


def generate_crossval_splits(data_path, dev_size=300, use_dev_set=False):
    """
    Inputs:
        data_path - Path to the directory in which train.jsonl and dev_seen.jsonl are.
        dev_size - Size of the development set inside the cross validation splits
        use_dev_set - If True, we will incorporate half of the dev_seen set into the training of each fold. The other half is used for testing. 
    """    
    random.seed(42)
    np.random.seed(42)
    data_list, dev_list = [], []
    for filepath in ["train.jsonl", "dev_seen.jsonl"]:
        assert os.path.isfile(os.path.join(data_path, filepath)), "Tried to create cross validation splits, but file could not be found at %s" % os.path.join(data_path, filepath)
        with open(os.path.join(data_path, filepath), "r") as f:
            json_list = f.readlines()
        json_list = [json.loads(json_str) for json_str in json_list]
        if filepath == "dev_seen.jsonl" and use_dev_set:
            dev_list = json_list
        else:
            shuffle(json_list)
            data_list += json_list
    data_by_label = {l: [d for d in data_list if d['label']==l] for l in [0,1]}
    dev_by_label = {l: [d for d in dev_list if d['label']==l] for l in [0,1]}
    num_splits = min([len(data_by_label[l]) for l in data_by_label]) // dev_size

    if use_dev_set:
        # We want to concatenate half of the dev_seen set to each training fold
        # Thereby, we want to make sure that each data point in dev_seen occurs
        # the same number of times in the test and training set. This split is
        # performed in the code lines below
        dev_by_split = []
        full_dev_size = len(dev_list)
        half_dev_size = full_dev_size // 2
        counts = np.zeros(full_dev_size, dtype=np.float32) + int(math.ceil(num_splits / 2.0))

        # Find text confounders
        exmp_by_text = defaultdict(list)
        for idx, exmp in enumerate(dev_list):
            exmp_by_text[exmp['text']].append(idx)
        confounder_list = [np.array(v, dtype=np.int32) for k,v in exmp_by_text.items() if len(v) > 1]
        confounder_idxs = np.array([v for vl in confounder_list for v in vl], dtype=np.int32)
        logger.info("Number of confounders: %i (sum: %i)" % (len(confounder_list), confounder_idxs.shape[0]))

        for split_id in range(num_splits):
            split_counts = np.copy(counts)
            
            # Decide split for confounders
            conf_to_include = np.array([], dtype=np.int32)
            splits_left = num_splits - split_id
            for cl in confounder_list:
                conf_count = counts[cl[0]]
                if conf_count >= splits_left or \
                   np.random.choice(2, size=1, p=[(splits_left-conf_count)/splits_left, conf_count/splits_left]) == 1:
                   conf_to_include = np.concatenate([conf_to_include, cl])
                   counts[cl[0]] -= 1

            split_counts[confounder_idxs] = 0

            # First step is to look for data points that would need to sampled in every split from now on to have a similar number of occurences as others
            samples_required = np.where(split_counts >= (num_splits - split_id))[0]
            # -> For uneven number of folds, it can happen that we would need to sample more than possible. Thus, we take a random selection preferring very rare examples
            spots_left = half_dev_size - conf_to_include.shape[0]
            if samples_required.shape[0] > spots_left:
                np.random.shuffle(samples_required)
                samples_required = samples_required[np.argsort(counts[samples_required][::-1])]
                samples_required = samples_required[:spots_left]
            spots_left -= samples_required.shape[0]
            # Mask out the samples we already took
            split_counts[samples_required] = 0
            if split_counts.sum() == 0:
                samples = np.zeros((0,))
            else:
                # Randomly sample from the rest of the data points
                samples = np.random.choice(counts.shape[0], size=spots_left, replace=False, p=split_counts/split_counts.sum())
                counts[samples] = counts[samples] - 1
            counts[samples_required] = counts[samples_required] - 1
            samples = samples.tolist() + np.arange(counts.shape[0])[samples_required].tolist() + conf_to_include.tolist()
            dev_by_split.append(samples)
        # Data points to use for training in the splits is the inverted from the test
        train_by_split = [[i for i in range(len(dev_list)) if i not in d] for d in dev_by_split]

        dev_by_split = [[dev_list[idx] for idx in d] for d in dev_by_split]
        train_by_split = [[dev_list[idx] for idx in d] for d in train_by_split]
        label_avgs = [sum([d['label'] for d in dlist])*1.0/len(dlist) for dlist in dev_by_split]
        logger.info("Label averages in test set: %s" % str(label_avgs))
        logger.info("Test set lengths: %s" % str([len(d) for d in dev_by_split]))


    data_path = os.path.join(data_path, "crossval_%i%s" % (dev_size, "" if not use_dev_set else "_usedevtest"))
    os.makedirs(data_path, exist_ok=True)
    for split_id in range(num_splits):
        start, end = split_id*(dev_size//2), (split_id+1)*(dev_size//2)
        dev_set = data_by_label[0][start:end] + data_by_label[1][start:end]
        train_set = data_by_label[0][:start]+data_by_label[0][end:]+data_by_label[1][:start]+data_by_label[1][end:]
        if use_dev_set:
            train_set += train_by_split[split_id]
            export_jsonl(os.path.join(data_path, "dev_seen_%s.jsonl" % str(split_id).zfill(2)), dev_by_split[split_id])
        export_jsonl(os.path.join(data_path, "train_%s.jsonl" % str(split_id).zfill(2)), train_set)
        export_jsonl(os.path.join(data_path, "dev_%s.jsonl" % str(split_id).zfill(2)), dev_set)
        label_avg = sum([d['label'] for d in dev_set]) * 1.0 / len(dev_set)
        logger.info("Exported split %i with %4.2f%% hateful memes in validation set." % (split_id, 100.0*label_avg))
    

def export_jsonl(filepath, dict_list):
    s = "\n".join([json.dumps(d) for d in dict_list])
    with open(filepath, "w") as f:
        f.write(s)


def train_crossval(trainer_class, config, data_loader_funcs, num_folds=0, dev_size=300, use_dev_set=False):
    if num_folds == 0:
        config['train_loader'] = data_loader_funcs["train"](os.path.join(config['data_path'], 'train.jsonl'))
        config['val_loader'] = data_loader_funcs["val"](os.path.join(config['data_path'], 'dev_seen.jsonl'))

        if hasattr(config['train_loader'].dataset, "name"):
            logger.info("Training on %s" % config['train_loader'].dataset.name)
            logger.info("Validating on %s" % config['val_loader'].dataset.name)

        trainer = None
        try:
            trainer = trainer_class(config)
            trainer.train_main()
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt by user detected at iteration %i..." % ((trainer.iters + trainer.total_iters) if trainer is not None else -1))
            logger.info("Closing the tensorboard writer")
            config['writer'].close()
    else:
        crossval_path = os.path.join(config['data_path'], "crossval_%i%s" % (dev_size, "" if not use_dev_set else "_usedevtest"))
        if not os.path.isdir(crossval_path) or len(glob(os.path.join(crossval_path, "*.jsonl"))) == 0:
            logger.info("Creating cross-validation splits for dev size %i" % dev_size)
            generate_crossval_splits(config['data_path'], dev_size=dev_size, use_dev_set=use_dev_set)
        train_sets = sorted(glob(os.path.join(crossval_path, "train_??.jsonl")))
        dev_sets = sorted(glob(os.path.join(crossval_path, "dev_??.jsonl")))
        test_sets = sorted(glob(os.path.join(crossval_path, "dev_seen_??.jsonl")))
        assert len(train_sets) == len(dev_sets), "Something seems to be wrong regarding the folds. Found an inequal number of training and validation sets.\nTrain: %s\nVal: %s" % (str(train_sets), str(dev_sets))
        if num_folds == -1:
            num_folds = len(dev_sets)
        if use_dev_set:
            assert len(test_sets) >= num_folds, "Could not find enough test sets."
        base_model_name, base_model_extension = config['model_save_name'].rsplit(".",1)
        original_test_loaders = config['test_loader']
        if use_dev_set:
            original_test_loaders = [t for t in original_test_loaders if t.dataset.name != "dev_seen"]

        early_stop = False
        trainer = None
        val_metrics = []
        try:
            folds_to_run = min(num_folds, len(dev_sets))
            for fold_idx in range(folds_to_run):

                set_seed(config['seed']+fold_idx)
                
                logger.info("Starting fold %i of %i (0-%i)" % (fold_idx, folds_to_run, folds_to_run-1))
                logger.info("Using train dataset %s, validation dataset %s" % (train_sets[fold_idx], dev_sets[fold_idx]))
                config['train_loader'] = data_loader_funcs["train"](train_sets[fold_idx])
                config['val_loader'] = data_loader_funcs["val"](dev_sets[fold_idx])
                if use_dev_set and len(test_sets) > fold_idx:
                    config['test_loader'] = original_test_loaders + [data_loader_funcs["test"](test_sets[fold_idx])]
                else:
                    config['test_loader'] = original_test_loaders

                config['model_save_name'] = base_model_name + "_fold_%i."%fold_idx + base_model_extension

                trainer = trainer_class(config)
                fold_val_metrics, _ = trainer.train_main()
                val_metrics.append(fold_val_metrics)

        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt by user detected at iteration %i..." % ((trainer.iters + trainer.total_iters) if trainer is not None else -1))
            logger.warning("Stopping cross validation early during fold %i" % fold_idx)
            logger.info("Closing the tensorboard writer")
            config['writer'].close()
            early_stop = False

        if not early_stop and len(val_metrics) != 0:
            mean_scores = {key: mean([v[key] for v in val_metrics]) for key in val_metrics[0]}
            logger.info("Cross validation finished. Mean scores of validation folds:\n" + "\n".join(["%s: %s" % (key, ("%4.2f%%"%(100.0*mean_scores[key])) if key != "loss" else "%5.4f"%(mean_scores[key])) for key in mean_scores]))

            base_path = os.path.join(config['model_path'], base_model_name + "_fold_*")
            dev_names = sorted([t.dataset.name for t in config['test_loader'] if t.dataset.name.startswith("dev")])
            if len(dev_names) == 0:
                logger.warning("Skipping ensemble calculation as no predictions for a validation set could be found")
            else:
                if not use_dev_set:
                    logger.info("Using %s for optimizing ensemble" % dev_names[0])
                    dev_files = sorted(glob(base_path + "_%s_preds.csv" % dev_names[0]))
                    test_names = [t.dataset.name for t in config['test_loader'] if t.dataset.name != dev_names[0]]
                else:
                    dev_files = sorted(glob(base_path + "_dev_seen_??_preds.csv"))
                    test_names = [t.dataset.name for t in original_test_loaders]
                test_files = [sorted(glob(base_path + "_%s_preds.csv" % n)) for n in test_names]
                find_ensemble(dev_files=dev_files, test_files=test_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        help='Path to folder of the meme dataset')
    args = parser.parse_args()
    generate_crossval_splits(data_path=args.data_path, dev_size=300, use_dev_set=True)