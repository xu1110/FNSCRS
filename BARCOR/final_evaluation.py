from torch.utils.data import Dataset
from utils import sample_data
from collections import defaultdict
import os
import torch
import argparse
import math
import os
import sys
import time
import json
import numpy as np
import torch
# import wandb
# from accelerate import Accelerator
# from accelerate.utils import set_seed
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from transformers.trainer_pt_utils import get_parameter_names

from dataset import DatasetForRec, DatasetForRecAug
from dataloader_bart import BARTDataCollatorForRec
from kg_bart import KGForBART
from metric import RecMetric
from utils import load_jsonl_data, simple_collate
from model_bart_unified import BartUnifiedModel
from torch.nn import functional as F

def remove_repeated_items(preds, entities):
    preds_filtered = []
    for pred in preds:
        if pred not in entities:
            preds_filtered.append(pred)
    return preds_filtered

def remove_repeated_items_batch(preds_batch, entities_batch):
    assert len(preds_batch) == len(entities_batch)
    preds_filtered_batch = []
    for i in range(len(preds_batch)):
        preds_filtered = remove_repeated_items(preds_batch[i], entities_batch[i])
        preds_filtered_batch.append(preds_filtered)
    return preds_filtered_batch

class DatasetForRecRL(Dataset):
    def __init__(
        self, data_list , tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None
    ):
        super(DatasetForRecRL, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id
        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)

    def prepare_data(self, data_list):
        index = 0
        for data in data_list:
            if len(data['rec']) == 0:
                continue

            text_list = []
            turn_idx = 0
            for utt in data['context']:
                if utt != '':
                    text = ''
                    if turn_idx % 2 == 0:
                        text += 'User: '
                    else:
                        text += 'System: '
                    text += utt
                    text_list.append(text)
                turn_idx += 1
            context = f'{self.tokenizer.sep_token}'.join(text_list)

            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)


            data_dict = {
                'index': index,
                'raw_context': context,
                'context': context_ids,
                'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                'rec': [self.entity2id[rec] for rec in data['rec'] if rec in self.entity2id]#self.entity2id[rec],
            }
            self.data_list.append(data_dict)
            index += 1



    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)

class BARTDataCollatorForRec:
    def __init__(
        self, tokenizer, device=torch.device('cpu'), debug=False, use_amp=False,
        context_max_length=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

    def __call__(self, data_batch, mode = None):
        input_batch = defaultdict(list)
        label_batch = []
        weight_batch = []
        raw_context = []
        entity = []
        index = []
        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            if 'raw_context' in data.keys():
                raw_context.append(data['raw_context'])
            entity.append(data['entity'])
            index.append(data['index'])
            label_batch.append(data['rec'])
            if 'weight' not in data:
                weight_batch.append(1)
            else:
                weight_batch.append(data['weight'])

        batch = {'weight': torch.as_tensor(weight_batch, device=self.device)}

        input_batch = self.tokenizer.pad(
            input_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        # print(input_batch)
        input_batch['labels'] = label_batch
        
        if mode == 'rlhf':
            for k, v in input_batch.items():
                if not isinstance(v, torch.Tensor) and k != 'index' and k != 'labels':
                    input_batch[k] = torch.as_tensor(v, device=self.device)
        else:
            for k, v in input_batch.items():
                if not isinstance(v, torch.Tensor):
                    input_batch[k] = torch.as_tensor(v, device=self.device)

        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids

        batch['input'] = input_batch
        
        # print(batch)
        
        if mode == 'rlhf':
            return index, batch, raw_context, entity
        elif mode == 'infer':
            return batch, raw_context
        else:
            return batch

class RecMetricRemoved:
    def __init__(self, k_list=(1,5, 10,25, 50)):
        self.k_list = k_list

        self.metric = {}
        self.reset_metric()

    def evaluate(self, preds, labels, entities):
        for pred_list, label_list, entity in zip(preds, labels, entities):
            for label in label_list:
                if label in entity:
                    continue
                if label == -100:
                    continue
                for k in self.k_list:
                    self.metric[f'recall@{k}'] += self.compute_recall(pred_list, label, k)
                    self.metric[f'ndcg@{k}'] += self.compute_ndcg(pred_list, label, k)
                    self.metric[f'mrr@{k}'] += self.compute_mrr(pred_list, label, k)
                    for pred in pred_list[:k]:
                        if pred not in self.pred_dict[k]:
                            self.pred_dict[k][pred] = 1
                        else:
                            self.pred_dict[k][pred] += 1
                self.metric['count'] += 1
    
    def compute_entropy(self, pred_dict):
        distribution = []
        for k, v in pred_dict.items():
            distribution.append(v)
        distribution = np.array(distribution, dtype=np.float64)
        distribution /= distribution.sum()
        entropy = -np.sum(distribution * np.log2(distribution))
        return entropy
    
    def compute_recall(self, pred_list, label, k):
        return int(label in pred_list[:k])

    def compute_mrr(self, pred_list, label, k):
        if label in pred_list[:k]:
            label_rank = pred_list.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, pred_list, label, k):
        if label in pred_list[:k]:
            label_rank = pred_list.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        for metric in ['recall', 'ndcg', 'mrr', 'coverage', 'entropy']:
            for k in self.k_list:
                self.metric[f'{metric}@{k}'] = 0
        self.pred_dict = {}
        for k in self.k_list:
            self.pred_dict[k] = {}
        self.metric['count'] = 0

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if k != 'count' and k!= 'coverage' and k!='entropy':
                report[k] = v / self.metric['count']
        
        for k in self.k_list:
            report[f'coverage@{k}'] = len(self.pred_dict[k])
            report[f'entropy@{k}'] = self.compute_entropy(self.pred_dict[k])
        return report

class RecMetricNotRemoved:
    def __init__(self, k_list=(1,5, 10,25, 50)):
        self.k_list = k_list

        self.metric = {}
        self.reset_metric()

    def evaluate(self, preds, labels, entities):
        for pred_list, label_list, entity in zip(preds, labels, entities):
            for label in label_list:
                if label == -100:
                    continue
                for k in self.k_list:
                    self.metric[f'recall@{k}'] += self.compute_recall(pred_list, label, k)
                    self.metric[f'ndcg@{k}'] += self.compute_ndcg(pred_list, label, k)
                    self.metric[f'mrr@{k}'] += self.compute_mrr(pred_list, label, k)
                self.metric['count'] += 1

    def compute_recall(self, pred_list, label, k):
        return int(label in pred_list[:k])

    def compute_mrr(self, pred_list, label, k):
        if label in pred_list[:k]:
            label_rank = pred_list.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, pred_list, label, k):
        if label in pred_list[:k]:
            label_rank = pred_list.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        for metric in ['recall', 'ndcg', 'mrr']:
            for k in self.k_list:
                self.metric[f'{metric}@{k}'] = 0
        self.metric['count'] = 0

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if k != 'count':
                report[k] = v / self.metric['count']
        return report


class FinalEvaluator():
    def __init__(self, model_id, args):
        self.args = args
        self.device = 'cuda'
        self.kg = KGForBART(kg_dataset=args.kg_dataset, debug=args.debug).get_kg_info()
        self.item_ids = torch.as_tensor(self.kg['item_ids'], device=self.device)
        # model
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.model = BartUnifiedModel.from_pretrained(model_id, torch_dtype=torch.float32, num_labels=self.kg['num_entities'], kg = self.kg, tokenizer = self.tokenizer).to(self.device)
        test_data_file = os.path.join(args.dataset, 'test_data_processed.jsonl')
        test_data_list = load_jsonl_data(test_data_file)
        test_dataset = DatasetForRecRL(
            data_list=test_data_list, entity2id=self.kg['entity2id'],
            tokenizer=self.tokenizer, context_max_length=args.context_max_length,
            debug=args.debug
        )
        self.dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=simple_collate,)
        self.data_collator = BARTDataCollatorForRec(device=self.device, debug=args.debug, use_amp=True,
        tokenizer=self.tokenizer, context_max_length=args.context_max_length)
        if 'redial' in args.dataset:
            self.dataset = 'redial'
        elif 'inspired' in args.dataset:
            self.dataset = 'inspired'
        elif 'reddit' in args.dataset:
            self.dataset = 'reddit'
        elif 'construct' in args.dataset:
            self.dataset = 'construct'
        self.evaluator_removed = RecMetricRemoved()
        self.evaluator_not_removed = RecMetricNotRemoved()
        
        
    @torch.no_grad()
    def evaluate(self, logger):
        self.model.eval()
        reports = {}
        # evaluate with repeated item removed
        for batch in tqdm(self.dataloader):
            index, batch, raw_context, entities = self.data_collator(batch, 'rlhf')
            # print(batch)
            labels = batch['input'].pop('labels')
            outputs = self.model.forward_rec(**batch['input'])
            # loss_list.append(float(outputs['loss']))
            logits = outputs['logits'][:, self.item_ids]
            ranks = torch.topk(logits, k=100, dim=-1).indices
            ranks = ranks.to(self.item_ids.device)
            preds = self.item_ids[ranks].tolist()
            self.evaluator_not_removed.evaluate(preds, labels, entities)
            preds_filtered = remove_repeated_items_batch(preds, entities)     
            self.evaluator_removed.evaluate(preds_filtered, labels, entities)
        
        reports['offline_metrics_removed'] = self.evaluator_removed.report()
        reports['offline_metrics_not_removed'] = self.evaluator_not_removed.report()
        for k, v in reports.items():
            logger.info(f'{k}: {v}')
        return reports
            
if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--log_dir", type=str, default='redial_rec')
        parser.add_argument("--infer_dir", type=str, default='redial_rec')
        parser.add_argument("--debug", action='store_true', help="Debug mode.")
        parser.add_argument("--dataset", type=str, default='./data/redial_rec', help="A file containing all data.")
        parser.add_argument("--kg_dataset", type=str, default='./kg_data/redial')
        parser.add_argument("--tokenizer", type=str, default='facebook/bart-base')#
        parser.add_argument("--context_max_length", type=int, default=160)
        parser.add_argument('--num_workers', type=int, default=0)
        # model
        parser.add_argument("--model", type=str, default='redial_rec/best')
        # optim
        parser.add_argument("--per_device_eval_batch_size", type=int, default=32,
                            help="Batch size (per device) for the evaluation dataloader.")
        parser.add_argument('--fp16', action='store_true')
        parser.add_argument('--repeated_item_removed', action='store_true')

        args = parser.parse_args()
        return args
    
    args = parse_args()
    logger.remove()
    # logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(args.log_dir, level='DEBUG')
    # logger.info(accelerator.state)
    config = vars(args)
    final_evaluator = FinalEvaluator(model_id=args.model, args=args)
    reports = final_evaluator.evaluate(logger=logger)
    print(reports)
