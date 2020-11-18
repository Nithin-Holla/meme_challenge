import time
import datetime
import shutil
import random
import sys
import os
import json
import re
import numpy as np
from statistics import mean, stdev
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from collections import defaultdict

from data.metrics import standard_metrics, find_optimal_threshold
from utils.optim_utils import get_optimizer
from utils.utils import calc_elapsed_time, print_stats, print_test_stats, log_tensorboard, set_seed, get_device
from utils.save import ModelSaver
from utils.logger import LOGGER


class TrainerTemplate():


    def __init__(self, config):
        self.preds_list, self.probs_list, self.labels_list, self.loss_list, self.short_loss_list, self.id_list = [], [], [], [], [], []
        self.best_val_metrics, self.train_metrics = defaultdict(int), {}
        self.best_auc = 0
        self.not_improved = 0
        self.best_val_loss = 1000
        self.total_iters = 0
        self.terminate_training = False
        self.model_file = os.path.join(config['model_path'], config['model_save_name'])
        self.pretrained_model_file = None
        if config['pretrained_model_file'] is not None:
            self.pretrained_model_file = os.path.join(config['model_path'], config['pretrained_model_file'])
        self.start_epoch = 1
        self.config = config
        self.device = get_device()

        if not isinstance(self.config['test_loader'], list):
            self.config['test_loader'] = [self.config['test_loader']]

        # Initialize the model, optimizer and loss function
        self.init_training_params()

        


    def init_training_params(self):
        self.init_model()
        self.model_saver = ModelSaver(self.model_file)

        if self.config['parallel_computing']:
            self.model = nn.DataParallel(self.model)

        self.init_optimizer()
        self.init_scheduler()

        if self.config['loss_func'] == 'bce_logits':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.config['pos_wt']]).to(self.device))
        elif self.config['loss_func'] == 'bce':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        

    def init_scheduler(self):
        if self.config['scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['lr_decay_step'], gamma=self.config['lr_decay_factor'])
        elif self.config['scheduler'] == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10, 15, 25, 40], gamma=self.config['lr_decay_factor'])
        elif self.config['scheduler'] == 'warmup':
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config['warmup_steps'], 
                                                             num_training_steps=len(self.config['train_loader']) * self.config['max_epoch'])
        elif self.config['scheduler'] == 'warmup_cosine':
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config['warmup_steps'], 
                                                             num_training_steps=len(self.config['train_loader']) * self.config['max_epoch'])


    def init_optimizer(self):
        self.optimizer = get_optimizer(self.model, self.config)


    def average_gradients(self, steps):
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = param.grad / steps


    def calculate_loss(self, preds, batch_label, grad_step):
        if self.config['loss_func'] == 'bce':
            preds = torch.sigmoid(preds)
        preds = preds.squeeze(1).to(self.device) if self.config['loss_func'] == 'bce_logits' else preds.to(self.device)  
        loss = self.criterion(preds, batch_label.to(self.device) if self.config['loss_func']=='ce' else batch_label.float().to(self.device))

        if grad_step and self.iters % self.config['gradient_accumulation'] == 0:
            loss.backward()
            self.average_gradients(steps=self.config['gradient_accumulation'])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        elif grad_step:
            loss.backward()

        if self.config['loss_func'] == 'bce':
            probs = preds
            preds = (preds > 0.5).type(torch.FloatTensor)
        elif self.config['loss_func'] == 'ce':
            probs = F.softmax(preds, dim=1)
            preds = torch.argmax(probs, dim=1)
        elif self.config['loss_func'] == 'bce_logits':
            probs = torch.sigmoid(preds)
            preds = (probs > 0.5).type(torch.FloatTensor)

        self.probs_list.append(probs.cpu().detach().numpy())
        self.preds_list.append(preds.cpu().detach().numpy())
        self.labels_list.append(batch_label.cpu().detach().numpy())
        self.loss_list.append(loss.detach().item())
        if grad_step:
            self.short_loss_list.append(loss.detach().item())




    def eval_model(self, test=False, test_idx=0):
        self.model.eval()
        self.preds_list, self.probs_list, self.labels_list, self.loss_list, self.id_list = [], [], [], [], []
        batch_loader = self.config['val_loader'] if not test else self.config['test_loader'][test_idx]
        with torch.no_grad():
            for iters, batch in enumerate(batch_loader):
                batch = self.batch_to_device(batch)
                if batch_loader.dataset.return_ids:
                    self.id_list.append(batch['ids'])
                self.eval_iter_step(iters, batch, test=test)

            self.probs_list = [prob for batch_prob in self.probs_list for prob in batch_prob]
            self.preds_list = [pred for batch_pred in self.preds_list for pred in batch_pred]
            self.labels_list = [label for batch_labels in self.labels_list for label in batch_labels]
            self.id_list = [data_id for batch_id in self.id_list for data_id in batch_id]

            val_loss = sum(self.loss_list)/len(self.loss_list)
            eval_metrics = standard_metrics(torch.tensor(self.probs_list), torch.tensor(self.labels_list), add_optimal_acc=True)
            # if test:
            # 	print(classification_report(np.array(self.labels_list), np.array(self.preds_list)))

        return eval_metrics, val_loss




    @torch.no_grad()
    def export_test_predictions(self, test_idx=0, threshold=0.5):
        self.model.eval()

        ## Step 2: Run model on the test set (no loss!)
        # Ensure that ids are actually returned
        assert self.config['test_loader'][test_idx].dataset.return_ids, "Can only export test results if the IDs are returned in the test dataset." 
        test_name = self.config['test_loader'][test_idx].dataset.name
        
        prob_list = []
        id_list = []
        for iters, batch in enumerate(self.config['test_loader'][test_idx]):
            batch = self.batch_to_device(batch)
            id_list.append(batch['ids'].cpu())
            probs = self.test_iter_step(batch)
            if self.config['loss_func'] == 'bce_logits':
                probs = torch.sigmoid(probs)
            prob_list.append(probs.detach().cpu())

        probs = torch.cat(prob_list, dim=0)
        ids = torch.cat(id_list, dim=0)
        preds = (probs > threshold).long()

        ## Step 3: Export predictions 
        self._export_preds(ids, probs, preds, file_postfix="_%s_preds.csv" % test_name)

        LOGGER.info("Finished export of test predictions")


    @torch.no_grad()
    def export_val_predictions(self, test=False, test_idx=0, threshold=0.5):
        batch_loader = self.config['val_loader'] if not test else self.config['test_loader'][test_idx]
        test_name = batch_loader.dataset.name
        LOGGER.info("Exporting %s predictions..." % (test_name))
        self.model.eval()

        ## Step 1: Find the optimal threshold on validation set
        _, _ = self.eval_model(test=test,test_idx=test_idx)
        val_probs = torch.tensor(self.probs_list)
        val_labels = torch.tensor(self.labels_list)
        if len(self.id_list) != 0:
            val_ids = torch.tensor(self.id_list)
        else:
            val_ids = torch.zeros_like(val_labels)-1
        val_preds = (val_probs > threshold).long()

        self._export_preds(val_ids, val_probs, val_preds, labels=val_labels, file_postfix="_%s_preds.csv" % test_name)

        LOGGER.info("Finished export of %s predictions" % test_name)


    def _export_preds(self, ids, probs, preds, labels=None, file_postfix="_preds.csv"):
        file_string = "id,proba,label%s\n" % (",gt" if labels is not None else "")
        for i in range(ids.shape[0]):
            file_string += "%i,%f,%i" % (ids[i].item(), probs[i].item(), preds[i].item())
            if labels is not None:
                file_string += ",%i" % labels[i].item()
            file_string += "\n"
        filepath = os.path.join(self.config['model_path'], self.config['model_save_name'].rsplit(".",1)[0] + file_postfix)
        with open(filepath, "w") as f:
            f.write(file_string)



    def check_early_stopping(self):
        self.this_metric = self.val_loss if self.config['optimize_for'] == 'loss' else self.val_metrics[self.config['optimize_for']]
        self.current_best = self.best_val_loss if self.config['optimize_for'] == 'loss' else self.best_val_metrics[self.config['optimize_for']]

        new_best = self.this_metric < self.current_best if self.config['optimize_for'] == 'loss' else self.this_metric > self.current_best
        if new_best:
            LOGGER.info("New High Score! Saving model...")
            self.best_val_metrics = self.val_metrics
            self.best_val_loss = self.val_loss
            if not self.config["no_model_checkpoints"]:
                self.model_saver.save(self.model)

        ### Stopping Criteria based on patience and change-in-metric-threshold ###
        diff = self.current_best - self.this_metric if self.config['optimize_for'] == 'loss' else self.this_metric - self.current_best
        if diff < self.config['early_stop_thresh']:
            self.not_improved += 1
            if self.not_improved >= self.config['patience']:
                self.terminate_training = True
        else:
            self.not_improved = 0
        LOGGER.info("current patience: {}".format(self.not_improved))




    def train_epoch_step(self):
        self.model.train()
        lr = self.scheduler.get_last_lr()
        self.total_iters += self.iters + 1
        self.probs_list = [pred for batch_pred in self.probs_list for pred in batch_pred]
        self.labels_list = [label for batch_labels in self.labels_list for label in batch_labels]

        # Evaluate on train set
        self.train_metrics = standard_metrics(torch.tensor(self.probs_list), torch.tensor(self.labels_list), add_optimal_acc=True)
        log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters, self.total_iters,
                        self.loss_list, self.train_metrics, lr[0], loss_only=False, val=False)
        self.train_loss = self.loss_list[:]

        # Evaluate on dev set
        val_time = time.time()
        self.val_metrics, self.val_loss = self.eval_model()
        self.config['writer'].add_scalar("Stats/time_validation", time.time() - val_time, self.total_iters)

        # print stats
        print_stats(self.config, self.epoch, self.train_metrics,
                    self.train_loss, self.val_metrics, self.val_loss, self.start, lr[0])

        # log validation stats in tensorboard
        log_tensorboard(self.config, self.config['writer'], self.model, self.epoch, self.iters,
                        self.total_iters, self.val_loss, self.val_metrics, lr[0], loss_only=False, val=True)

        # Check for early stopping criteria
        self.check_early_stopping()
        self.probs_list = []
        self.preds_list = []
        self.labels_list = []
        self.loss_list = []
        self.id_list = []

        self.train_loss = sum(self.train_loss)/len(self.train_loss)
        del self.val_metrics
        del self.val_loss




    def end_training(self):
        # Termination message
        print("\n" + "-"*100)
        if self.terminate_training:
            LOGGER.info("Training terminated early because the Validation {} did not improve for   {}   epochs" .format(self.config['optimize_for'], self.config['patience']))
        else:
            LOGGER.info("Maximum epochs of {} reached. Finished training !!".format(self.config['max_epoch']))

        print_test_stats(self.best_val_metrics, test=False)

        print("-"*50 + "\n\t\tEvaluating on test set\n" + "-"*50)
        if not self.config["no_model_checkpoints"]:
            if os.path.isfile(self.model_file):
                self.load_model()
                self.model.to(self.device)
            else:
                raise ValueError("No Saved model state_dict found for the chosen model...!!! \nAborting evaluation on test set...".format(self.config['model_name']))

            self.export_val_predictions() # Runs evaluation, no need to run it again here
            val_probs = torch.tensor(self.probs_list)
            val_labels = torch.tensor(self.labels_list)
            threshold = find_optimal_threshold(val_probs, val_labels, metric="accuracy", show_plot=False)
            best_val_metrics = standard_metrics(val_probs, val_labels, threshold=threshold, add_aucroc=False)
            LOGGER.info("Optimal threshold on validation dataset: %.4f (accuracy=%4.2f%%)" % (threshold, 100.0*best_val_metrics["accuracy"]))

            ## Testing is in the standard form not possible, as we do not have any labels (gives an error in standard_metrics)
            ## Instead, we should write out the predictions in the form of the leaderboard
            self.test_metrics = dict()
            for test_idx in range(len(self.config['test_loader'])):
                test_name = self.config['test_loader'][test_idx].dataset.name
                LOGGER.info("Export and testing on %s..." % test_name)
                if hasattr(self.config['test_loader'][test_idx].dataset, "data") and \
                   hasattr(self.config['test_loader'][test_idx].dataset.data, "labels") and \
                   self.config['test_loader'][test_idx].dataset.data.labels[0] == -1:## Step 1: Find the optimal threshold on validation set
                    self.export_test_predictions(test_idx=test_idx, threshold=threshold)
                    self.test_metrics[test_name] = dict()
                else:
                    test_idx_metrics, _ = self.eval_model(test=True, test_idx=test_idx)
                    self.test_metrics[test_name] = test_idx_metrics
                    print_test_stats(test_idx_metrics, test=True)
                    self.export_val_predictions(test=True, test_idx=test_idx, threshold=threshold)
        else:
            LOGGER.info("No model checkpoints were saved. Hence, testing will be skipped.")
            self.test_metrics = dict()

        self.export_metrics()

        self.config['writer'].close()

        if self.config['remove_checkpoints']:
            LOGGER.info("Removing checkpoint %s..." % self.model_file)
            os.remove(self.model_file)




    def export_metrics(self):
        metric_export_file = os.path.join(self.config['model_path'], self.config['model_save_name'].rsplit(".",1)[0] + "_metrics.json")
        metric_dict = {}
        metric_dict["dev"] = self.best_val_metrics
        metric_dict["dev"]["loss"] = self.best_val_loss
        metric_dict["train"] = self.train_metrics
        metric_dict["train"]["loss"] = sum(self.train_loss)/len(self.train_loss) if isinstance(self.train_loss, list) else self.train_loss
        if hasattr(self, "test_metrics") and len(self.test_metrics) > 0:
            metric_dict["test"] = self.test_metrics

        with open(metric_export_file, "w") as f:
            json.dump(metric_dict, f, indent=4)




    def train_main(self, cache=False):
        print("\n\n" + "="*100 + "\n\t\t\t\t\t Training Network\n" + "="*100)

        self.start = time.time()
        print("\nBeginning training at:  {} \n".format(datetime.datetime.now()))

        self.model.to(self.device)

        for self.epoch in range(self.start_epoch, self.config['max_epoch']+1):
            train_times = []
            for self.iters, self.batch in enumerate(self.config['train_loader']):
                self.model.train()

                iter_time = time.time()
                self.batch = self.batch_to_device(self.batch)
                self.train_iter_step()
                train_times.append(time.time() - iter_time)

                # Loss only
                if (self.total_iters+self.iters+1) % self.config['log_every'] == 0:
                    ## Uncomment line below for debugging
                    if self.config['debug']:
                        LOGGER.info("Logging tensorboard at step %i with %i values" % (self.iters+self.total_iters+1, len(self.short_loss_list)))
                    log_tensorboard(self.config, self.config['writer'], self.model, self.epoch,
                                    self.iters, self.total_iters, self.short_loss_list, loss_only=True, val=False)
                    self.config['writer'].add_scalar('Stats/time_per_train_iter', mean(train_times), (self.iters+self.total_iters+1))
                    self.config['writer'].add_scalar('Stats/learning_rate', self.scheduler.get_last_lr()[0], (self.iters+self.total_iters+1))
                    train_times = []
                    self.short_loss_list = []
            self.train_epoch_step()

            if self.terminate_training:
                break

        self.end_training()
        return self.best_val_metrics, self.test_metrics


    def batch_to_device(self, batch):
        batch = {k : (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
        return batch


    def init_model(self):
        raise NotImplementedError


    def load_model(self):
        raise NotImplementedError


    def train_iter_step(self):
        raise NotImplementedError


    def eval_iter_step(self, iters, batch, test):
        raise NotImplementedError


    def test_iter_step(self, batch):
        raise NotImplementedError




    @staticmethod
    def add_default_argparse(parser, defaults=dict()):
        # Required Paths
        parser.add_argument('--data_path', type=str, default='./dataset',
                            help='path to dataset folder that contains the processed data files')
        parser.add_argument('--model_path', type=str, default='./model_checkpoints',
                            help='Directory for saving trained model checkpoints')
        parser.add_argument('--vis_path', type=str, default='./vis_checkpoints',
                            help='Directory for saving tensorboard checkpoints')
        parser.add_argument("--model_save_name", type=str, default='best_model.pt',
                            help='saved model name')
        parser.add_argument("--no_model_checkpoints", action="store_true",
                            help='If selected, no model checkpoints will be created, and no testing performed (for gridsearches etc.)')
        parser.add_argument("--remove_checkpoints", action="store_true",
                            help='If selected, model checkpoints will be deleted after finishing testing.')
        parser.add_argument('--debug', action="store_true",
                            help='This option is intended for tests on local machines, and more output.')

        # Load pretrained model
        parser.add_argument('--pretrained_model_file', type=str, help='Name of the pretrained model')

        ## Training parameters

        # Named parameters
        parser.add_argument('--optimizer', type=str, default=defaults.get('optimizer', 'adam'),
                        help='Optimizer to use for training: adam / adamx / adamw')
        ## Not sure whether we should have this here. For a multi-task setup, we need our own loss functions
        parser.add_argument('--loss_func', type=str, default=defaults.get('loss_func', 'bce_logits'),
                            help='Loss function to use for optimization: bce / bce_logits / ce')
        parser.add_argument('--optimize_for', type=str, default=defaults.get('optimize_for', 'aucroc'),
                            help='Optimize for what measure during training and early stopping: loss / F1 / aucroc / accuracy')
        parser.add_argument('--scheduler', type=str, default=defaults.get('scheduler', 'warmup_cosine'),
                            help='The type of lr scheduler to use anneal learning rate: step/multi_step/warmup/warmp_cosine')

        # Numerical parameters
        parser.add_argument('--confounder_repeat', type=int, default=defaults.get('confounder_repeat', 1),
                            help="Factor with which we should repeat the (hard) text confounding examples during training.")
        parser.add_argument('--object_conf_thresh', type=float, default=defaults.get('object_conf_thresh', 0.0),
                            help="Confidence threshold for object bounding boxes. Boxes with lower confidence are ignored.")
        parser.add_argument('--num_folds', type=int, default=defaults.get('num_folds', 0),
                              help='Number of folds to use during training. 0 means the default split, -1 means all splits')
        parser.add_argument('--crossval_dev_size', type=int, default=defaults.get('crossval_dev_size', 300),
                              help='Size of the development folds used in cross validation. Default: 300 (150 positive, 150 negative).')
        parser.add_argument('--crossval_use_dev', action="store_true",
                            help='If selected, the dev_seen set is incorporated into the cross validation splits.')
        parser.add_argument('--beta1', type=float, default=defaults.get('beta1', 0.9),
                              help='beta1 parameter in Adam optimizer')
        parser.add_argument('--beta2', type=float, default=defaults.get('beta2', 0.999),
                              help='beta2 parameter in Adam optimizer')
        parser.add_argument('--batch_size', type=int, default=defaults.get('batch_size', 8),
                              help='batch size for training')
        parser.add_argument('--num_workers', type=int, default=defaults.get('num_workers', 0),
                              help='Number of workers to start per dataset')
        parser.add_argument('--gradient_accumulation', type=int, default=defaults.get('gradient_accumulation', 1),
                            help='No. of update steps to accumulate before performing backward pass')
        parser.add_argument('--max_grad_norm', type=int, default=defaults.get('max_grad_norm', 5),
                            help='max gradient norm for gradient clipping')
        parser.add_argument('--pos_wt', type=float, default=defaults.get('pos_wt', 1),
                              help='Loss reweighting for the positive class to deal with class imbalance')
        parser.add_argument('--lr', type=float, default=defaults.get('lr', 1e-4),
                          help='Learning rate for training')
        parser.add_argument('--warmup_steps', type=int, default=defaults.get('warmup_steps', 50),
                              help='No. of steps to perform linear lr warmup for')
        parser.add_argument('--weight_decay', type=float, default=defaults.get('weight_decay', 1e-3),
                            help='weight decay for optimizer')
        parser.add_argument('--max_epoch', type=int, default=defaults.get('max_epoch', 20),
                            help='Max epochs to train for')
        parser.add_argument('--lr_decay_step', type=float, default=defaults.get('lr_decay_step', 3),
                            help='No. of epochs after which learning rate should be decreased')
        parser.add_argument('--lr_decay_factor', type=float, default=defaults.get('lr_decay_factor', 0.8),
                            help='Decay the learning rate of the optimizer by this multiplicative amount')
        parser.add_argument('--patience', type=float, default=defaults.get('patience', 5),
                            help='Patience no. of epochs for early stopping')
        parser.add_argument('--early_stop_thresh', type=float, default=defaults.get('early_stop_thresh', 1e-3),
                            help='Patience no. of epochs for early stopping')
        parser.add_argument('--seed', type=int, default=defaults.get('seed', 42),
                            help='set seed for reproducability')
        parser.add_argument('--log_every', type=int, default=defaults.get('log_every', 2000),
                            help='Log stats in Tensorboard every x iterations (not epochs) of training')

        # Options params
        parser.add_argument('--parallel_computing', type=bool, default=defaults.get('parallel_computing', False),
                            help='To run the model on multiple GPUs')




    @staticmethod
    def preprocess_args(config):
        config['device'] = get_device()
        config['n_classes'] = 2 if config['loss_func'] == 'ce' else 1

        # Check all provided paths:
        if not os.path.exists(config['data_path']):
            raise ValueError("[!] ERROR: Dataset path does not exist")
        else:
            LOGGER.info("Data path checked..")
        if not os.path.exists(config['model_path']):
            LOGGER.warning("Creating checkpoint path for saved models at:  {}\n".format(config['model_path']))
            os.makedirs(config['model_path'])
        else:
            LOGGER.info("Model save path checked..")
        if 'config' in config:
            if not os.path.isfile(config['config']):
                raise ValueError("[!] ERROR: config JSON path does not exist")
            else:
                LOGGER.info("config JSON path checked..")
        if not os.path.exists(config['vis_path']):
            LOGGER.warning("Creating checkpoint path for Tensorboard visualizations at:  {}\n".format(config['vis_path']))
            os.makedirs(config['vis_path'])
        else:
            LOGGER.info("Tensorboard Visualization path checked..")
            LOGGER.info("Cleaning Visualization path of older tensorboard files...\n")
            # shutil.rmtree(config['vis_path'])

        # Print args
        print("\n" + "x"*50 + "\n\nRunning training with the following parameters: \n")
        for key, value in config.items():
            if not key.endswith('transf'):
                print(key + ' : ' + str(value))
        print("\n" + "x"*50)

        # config['vis_path'] = os.path.join(config['vis_path'], '{}_conf{}'.format(config['pretrained_model_file'], config['confounder_repeat']))
        config['writer'] = SummaryWriter(config['vis_path'])

        set_seed(config['seed'])
        return config


