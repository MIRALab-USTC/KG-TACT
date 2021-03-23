import statistics
import timeit
import os
import logging
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from sklearn import metrics


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.critic = ['auc', 'auc_pr', 'mrr']
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=0.9, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        if self.params.loss:
            logging.info('using abs loss!')

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0

        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())

        for b_idx, batch in enumerate(dataloader):
            # data_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
            num_neg_rel = min(self.params.num_neg_samples_per_link, self.params.num_rels - 1)
            ###############
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()
            score_pos, _ = self.graph_classifier(data_pos)  # torch.Size([16, 1]) torch.Size([16, 8, 1])
            score_neg, _ = self.graph_classifier(data_neg)
            #################
            # score_pos, score_neg = self.graph_classifier(data_pos)
            # # score_neg = self.graph_classifier(data_neg)
            if self.params.loss == 1:
                # score_pos, score_neg = F.relu(score_pos), F.relu(score_neg)
                # loss = torch.abs(torch.sum(score_neg) + torch.sum(torch.clamp(self.params.margin - score_pos, min=0)))
                loss = torch.abs(torch.sum(torch.sum(score_neg, dim=1) + torch.clamp(self.params.margin - score_pos, min=0)))
            else:
                loss = self.criterion(score_pos, score_neg.mean(dim=1), torch.Tensor([1]).to(device=self.params.device))

            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                all_scores += score_pos.squeeze().detach().cpu().tolist() + score_neg.view(-1,
                                                                                           1).squeeze().detach().cpu().tolist()

                all_labels += targets_pos.tolist() + [0] * (num_neg_rel * len(targets_pos.tolist()))
                total_loss += loss

            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info('Performance: ' + str(result) + 'in ' + str(time.time() - tic))

                if result[self.critic[self.params.critic]] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result[self.critic[self.params.critic]]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(
                            f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result[self.critic[self.params.critic]]

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm

    def train(self):

        log_dir = os.path.join('logs', self.params.experiment_name)
        # writer = SummaryWriter(log_dir=log_dir)
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            self.params.epoch = epoch
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            # writer.add_scalar('Loss', loss, epoch)
            # writer.add_scalar('AUC', auc, epoch)
            # writer.add_scalar('AUC-PR', auc_pr, epoch)
            # writer.add_scalar('weight_norm', weight_norm, epoch)
            time_elapsed = time.time() - time_start
            logging.info(
                f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))
        # writer.close()

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir,
                                                       'best_graph_classifier.pth'))
        logging.info(f'Better models found w.r.t {self.critic[self.params.critic]}. Saved it!')
