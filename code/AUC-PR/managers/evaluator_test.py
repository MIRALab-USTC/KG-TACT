import os
import numpy as np
import torch
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()

        # num_neg_rel = min(self.params.num_neg_samples_per_link, self.params.num_rels - 1)
        # for testing
        num_neg_rel = self.params.num_neg_samples_per_link

        with torch.no_grad():
            ranks = []
            for b_idx, batch in enumerate(dataloader):
                # data_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # for testing
                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch,
                                                                                                self.params.device)
                # score_pos, score_neg = self.graph_classifier(data_pos)  # torch.Size([16, 1]) torch.Size([16, 8, 1])
                score_pos, _ = self.graph_classifier(data_pos)  # torch.Size([16, 1]) torch.Size([16, 8, 1])
                score_neg, _ = self.graph_classifier(data_neg)

                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()  # list [#16]
                neg_scores += score_neg.view(-1, 1).squeeze(1).detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += [0] * (num_neg_rel * len(targets_pos.tolist()))
                # ####### mrr
                # scores = torch.cat((score_pos, score_neg.squeeze(dim=2)), dim=1)  # torch.Size([16, 9])
                scores = torch.cat([score_pos, score_neg.view_as(score_pos)], dim=1)  # torch.Size([16, 2])
                scores = torch.softmax(scores, dim=1)
                scores = scores.detach().cpu().numpy()
                rank = np.argwhere(np.argsort(scores, axis=1)[:, ::-1] == 0)[:, 1] + 1
                ranks += rank.tolist()

                # for testing
                # scores = torch.cat((score_pos, score_neg.view(-1, 1)), dim=0)
                # scores = torch.softmax(scores, dim=0).squeeze(1)
                # scores = scores.detach().cpu().numpy()
                # rank = np.argwhere(np.argsort(scores)[::-1] == 0) + 1
                # ranks.append(rank)

        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)
        mrr = np.mean(1.0 / np.array(ranks)).item()

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir,
                                                  '../../data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir,
                                         '../../data/{}/grail_{}_predictions.txt'.format(self.params.dataset,
                                                                                   self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir,
                                                  '../../data/{}/neg_{}_0.txt'.format(self.params.dataset,
                                                                                self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir,
                                         '../../data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset,
                                                                                          self.data.file_name,
                                                                                          self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr, 'mrr': mrr}
