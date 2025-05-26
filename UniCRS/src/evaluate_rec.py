import math
import numpy as np
import torch


class RecEvaluator:
    def __init__(self, k_list=None, device=torch.device('cpu'), itemids=None):
        if k_list is None:
            k_list = [1,5, 10,25, 50]
        self.k_list = k_list
        self.device = device
        self.itemids = itemids
        self.metric = {}
        self.reset_metric()

    def compute_entropy(self, pred_dict):
        distribution = []
        for k, v in pred_dict.items():
            distribution.append(v)
        distribution = np.array(distribution, dtype=np.float64)
        distribution /= distribution.sum()
        entropy = -np.sum(distribution * np.log2(distribution))
        return entropy
    
    def evaluate(self, logits, labels):
        for logit, label in zip(logits, labels):
            for k in self.k_list:
                self.metric[f'recall@{k}'] += self.compute_recall(logit, label, k)
                self.metric[f'mrr@{k}'] += self.compute_mrr(logit, label, k)
                self.metric[f'ndcg@{k}'] += self.compute_ndcg(logit, label, k)
                if self.itemids != None:
                    for pred in logit[:k]:
                        if pred not in self.pred_dict[k]:
                            self.pred_dict[k][pred] = 1
                        else:
                            self.pred_dict[k][pred] += 1
            self.metric['count'] += 1

    def compute_recall(self, rank, label, k):
        return int(label in rank[:k])

    def compute_mrr(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        if self.itemids != None:
            for metric in ['recall', 'ndcg', 'mrr', 'coverage', 'entropy']:
                for k in self.k_list:
                    self.metric[f'{metric}@{k}'] = 0
            self.pred_dict = {}
            for k in self.k_list:
                self.pred_dict[k] = {}
            self.metric['count'] = 0
        else:
            for metric in ['recall', 'ndcg', 'mrr']:
                for k in self.k_list:
                    self.metric[f'{metric}@{k}'] = 0
            self.metric['count'] = 0

    def report(self):
        report = {}
        if self.itemids != None:
            for k in self.k_list:
                self.metric[f'coverage@{k}'] = len(self.pred_dict[k]) * self.metric['count']
                self.metric[f'entropy@{k}'] = self.compute_entropy(self.pred_dict[k]) * self.metric['count']
        
        for k, v in self.metric.items():
            report[k] = torch.tensor(v, device=self.device)[None]
        
        return report
