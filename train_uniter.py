import argparse
import os
import torch
from functools import partial
from torch.utils import data
from transformers import BertTokenizer

from model.meme_uniter import MemeUniter
from model.pretrain import UniterForPretraining
from utils.logger import LOGGER
from train_template import TrainerTemplate
from data.meme_dataset import MemeDataset, ConfounderSampler
from model.model import UniterModel, UniterConfig
from utils.const import IMG_DIM, IMG_LABEL_DIM
from utils.utils import get_gather_index, get_attention_mask
from utils.crossval import train_crossval


class TrainerUniter(TrainerTemplate):


    def init_model(self):
        if self.pretrained_model_file:
            checkpoint = torch.load(self.pretrained_model_file)
            LOGGER.info('Using pretrained UNITER base model {}'.format(self.pretrained_model_file))
            base_model = UniterForPretraining.from_pretrained(self.config['config'],
                                                              state_dict=checkpoint['model_state_dict'],
                                                              img_dim=IMG_DIM,
                                                              img_label_dim=IMG_LABEL_DIM)
            self.model = MemeUniter(uniter_model=base_model.uniter,
                                    hidden_size=base_model.uniter.config.hidden_size,
                                    n_classes=self.config['n_classes'])
        else:
            self.load_model()




    def load_model(self):
        # Load pretrained model
        if self.model_file:
            checkpoint = torch.load(self.model_file)
            LOGGER.info('Using UNITER model {}'.format(self.model_file))
        else:
            checkpoint = {}

        uniter_config = UniterConfig.from_json_file(self.config['config'])
        uniter_model = UniterModel(uniter_config, img_dim=IMG_DIM)

        self.model = MemeUniter(uniter_model=uniter_model,
                                hidden_size=uniter_model.config.hidden_size,
                                n_classes=self.config['n_classes'])
        self.model.load_state_dict(checkpoint['model_state_dict'])




    def eval_iter_step(self, iters, batch, test):
        # Forward pass
        preds = self.model(img_feat=batch['img_feat'], img_pos_feat=batch['img_pos_feat'], input_ids=batch['input_ids'],
                            position_ids=batch['position_ids'], attention_mask=batch['attn_mask'], gather_index=batch['gather_index'],
                            output_all_encoded_layers=False)
        self.calculate_loss(preds, batch['labels'], grad_step=False)



    def train_iter_step(self):
        # Forward pass
        self.preds = self.model(img_feat=self.batch['img_feat'], img_pos_feat=self.batch['img_pos_feat'], input_ids=self.batch['input_ids'],
                            position_ids=self.batch['position_ids'], attention_mask=self.batch['attn_mask'], gather_index=self.batch['gather_index'],
                            output_all_encoded_layers=False)
        self.calculate_loss(self.preds, self.batch['labels'], grad_step=True)



    def test_iter_step(self, batch):
        # Forward pass
        preds = self.model(img_feat=batch['img_feat'], img_pos_feat=batch['img_pos_feat'], input_ids=batch['input_ids'],
                            position_ids=batch['position_ids'], attention_mask=batch['attn_mask'], gather_index=batch['gather_index'],
                            output_all_encoded_layers=False)        
        return preds.squeeze()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    TrainerTemplate.add_default_argparse(parser)

    # Required Paths
    parser.add_argument('--config', type=str, default='./config/uniter-base.json',
                        help='JSON config file')
    parser.add_argument('--feature_path', type=str, default='./dataset/img_feats',
                        help='Path to image features')
    
    #### Pre-processing Params ####
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes (-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    #### Training Params ####

    # Numerical params
    parser.add_argument('--fc_dim', type=int, default=64,
                          help='dimen of FC layer"')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Standard dropout regularization')



    args, unparsed = parser.parse_known_args()
    config = args.__dict__
    config = TrainerTemplate.preprocess_args(config)

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokenizer_func = partial(tokenizer, max_length=config['max_txt_len'], padding='max_length',
                             truncation=True, return_tensors='pt', return_length=True)


    # Prepare the datasets and iterator for training and evaluation based on Glove or Elmo embeddings
    # train_dataset = MemeDataset(filepath=os.path.join(config['data_path'], 'train.jsonl'),
    #                             feature_dir=config['feature_path'],
    #                             preload_images=False, debug=True, text_padding=tokenizer_func)
    # val_dataset = MemeDataset(filepath=os.path.join(config['data_path'], 'dev_seen.jsonl'),
    #                           feature_dir=config['feature_path'],
    #                           preload_images=False, debug=True, text_padding=tokenizer_func)
    # test_dataset = MemeDataset(filepath=os.path.join(config['data_path'], 'test_seen.jsonl'),
    #                            feature_dir=config['feature_path'],
    #                            return_ids=True,
    #                            preload_images=False, debug=True, text_padding=tokenizer_func)
    
    # config['train_loader'] = data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=train_dataset.get_collate_fn(), shuffle=True, pin_memory=True)
    # config['val_loader'] = data.DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=val_dataset.get_collate_fn())
    # config['test_loader'] = data.DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=test_dataset.get_collate_fn())


    # try:
    #     trainer = TrainerUniter(config)
    #     trainer.train_main()
    # except KeyboardInterrupt:
    #     LOGGER.warning("Keyboard interrupt by user detected...\nClosing the tensorboard writer!")
    #     config['writer'].close()

    ## Cross validation (not tested!)
    def train_data_loader(train_file):
        train_dataset = MemeDataset(filepath=train_file,
                                    feature_dir=config['feature_path'],
                                    preload_images=False, debug=True, text_padding=tokenizer_func,
                                    confidence_threshold=config['object_conf_thresh'])
        return data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], 
                               collate_fn=train_dataset.get_collate_fn(), pin_memory=True, # shuffle is mutually exclusive with sampler. It is shuffled anyways
                               sampler=ConfounderSampler(train_dataset, repeat_factor=config["confounder_repeat"]))

    def val_data_loader(val_file):
        val_dataset = MemeDataset(filepath=val_file,
                                  feature_dir=config['feature_path'],
                                  preload_images=False, debug=True, text_padding=tokenizer_func,
                                  confidence_threshold=config['object_conf_thresh'])
        return data.DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=val_dataset.get_collate_fn())


    def test_data_loader(test_file):
        test_dataset = MemeDataset(filepath=test_file,
                                   feature_dir=config['feature_path'],
                                   return_ids=True,
                                   preload_images=False, debug=True, text_padding=tokenizer_func,
                                   confidence_threshold=config['object_conf_thresh'])
        return data.DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=test_dataset.get_collate_fn())

    config['test_loader'] = []
    for test_file in ['test_seen.jsonl', 'test_unseen.jsonl', 'dev_seen.jsonl', 'dev_unseen.jsonl']:
        config['test_loader'].append(test_data_loader(os.path.join(config['data_path'], test_file)))

    train_crossval(trainer_class=TrainerUniter,
                   config=config,
                   data_loader_funcs={"train": train_data_loader, "val": val_data_loader, "test": test_data_loader},
                   num_folds=config['num_folds'],
                   dev_size=config['crossval_dev_size'],
                   use_dev_set=config['crossval_use_dev'])