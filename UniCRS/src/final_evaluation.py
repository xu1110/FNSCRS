import argparse
import math
import os
import sys
import time
import json
import numpy as np
import torch
import transformers
# import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
import pandas as pd
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict, gpt2_chinese_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
from torch.nn import functional as F
import torch.nn as nn
import json
import os
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from utils import padded_tensor


class RecMetricRemoved:
    def __init__(self, k_list=(1,5, 10,25, 50)):
        self.k_list = k_list

        self.metric = {}
        self.reset_metric()

    def evaluate(self, preds, labels, entities):
        for pred_list, label_list, entity in zip(preds, labels, entities):
            # print(len(pred_list))
            # print(len(label_list))
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
        report['count'] = self.metric['count']
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


 
class CRSRecRLDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, entity2id, repeated_item_removed, language, debug=False,
        context_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,
        use_resp=False
    ):
        super(CRSRecRLDataset, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.repeated_item_removed = repeated_item_removed
        self.prompt_tokenizer = prompt_tokenizer
        self.use_resp = use_resp
        self.entity2id = entity2id
        self.context_max_length = context_max_length
        self.language = language
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        dataset_dir = os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        self.data = []
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]
            index = 0
            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:
                    continue
                if len(dialog['context']) == 1 and dialog['context'][0] == '':
                    continue

                context = ''
                prompt_context = ''
                
                for i, utt in enumerate(dialog['context']):
                    if utt == '':
                        continue
                    if i % 2 == 0:
                        context += 'User: ' if self.language == 'english' else '用户: '
                        prompt_context += 'User: ' if self.language == 'english' else '用户: '
                    else:
                        context += 'System: ' if self.language == 'english' else '系统: '
                        prompt_context += 'System: ' if self.language == 'english' else '系统: '
                    context += utt
                    context += self.tokenizer.eos_token
                    prompt_context += utt
                    prompt_context += self.prompt_tokenizer.sep_token

                if context == '':
                    continue
                if self.use_resp:
                    if i % 2 == 0:
                        resp = 'System: ' if self.language == 'english' else '系统: '
                    else:
                        resp = 'User: ' if self.language == 'english' else '用户: '
                    resp += dialog['resp']
                    context += resp + self.tokenizer.eos_token
                    prompt_context += resp + self.prompt_tokenizer.sep_token

                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.context_max_length:]

                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))
                prompt_ids = prompt_ids[-self.prompt_max_length:]
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)
                if len(dialog['rec']) != 0 and len(dialog['entity']) != 0:
                    data = {
                        'context': context_ids,
                        'entity': [self.entity2id[entityid] for entityid in dialog['entity'][-self.entity_max_length:] if entityid in self.entity2id],
                        'rec': [self.entity2id[rec] for rec in dialog['rec'] if rec in self.entity2id],
                        'prompt': prompt_ids,
                        'raw_context': context,
                        'index': index
                    }
                    self.data.append(data)


    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)

class CRSRecRLDataCollator:
    def __init__(
        self, tokenizer, device, pad_entity_id, use_amp=False, debug=False,
        context_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length

        self.pad_entity_id = pad_entity_id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        # self.rec_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('Recommend:'))

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []
        raw_context_batch=[]
        index_batch = []
        for data in data_batch:
            # input_ids = data['context'][-(self.context_max_length - len(self.rec_prompt_ids)):] + self.rec_prompt_ids
            input_ids = data['context']
            context_batch['input_ids'].append(input_ids)
            entity_batch.append(data['entity'])
            label_batch.append(data['rec'])
            prompt_batch['input_ids'].append(data['prompt'])
            raw_context = data['raw_context']
            raw_context_batch.append(raw_context)
            index_batch.append(data['index'])
        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length
        )
        context_batch['rec_labels'] = label_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor) and k != 'rec_labels':
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, padding=self.padding, max_length=self.prompt_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch_pad = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
        input_batch['entity'] = entity_batch_pad
        input_batch['raw_context'] = raw_context_batch
        input_batch['index'] = index_batch
        input_batch['mentioned_entities'] = entity_batch
        return input_batch



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

class FinalEvaluator():
    def __init__(self, model_id, args):
        self.args = args
        self.device = 'cuda'
        self.kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
        self.item_ids = torch.as_tensor(self.kg['item_ids'], device=self.device)
        # model
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if args.language == 'chinese':
            self.tokenizer.add_special_tokens(gpt2_chinese_tokens_dict)
        else:
            self.tokenizer.add_special_tokens(gpt2_special_tokens_dict)
        self.model = PromptGPT2forCRS.from_pretrained(args.model)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = self.model.to(self.device)
        if 'redial' in args.dataset:
            dataset = 'redial'
        elif 'inspired' in args.dataset:
            dataset = 'inspired'
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
        self.text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
        self.text_encoder = AutoModel.from_pretrained(args.text_encoder)
        self.text_encoder.resize_token_embeddings(len(self.text_tokenizer))
        self.text_encoder = self.text_encoder.to(self.device)

        self.prompt_encoder = KGPrompt(
            self.model.config.n_embd, self.text_encoder.config.hidden_size, self.model.config.n_head, self.model.config.n_layer, 2,
            n_entity=self.kg['num_entities'], num_relations=self.kg['num_relations'], num_bases=args.num_bases,
            edge_index=self.kg['edge_index'], edge_type=self.kg['edge_type'],
            n_prefix_rec=args.n_prefix_rec
        )
        self.prompt_encoder.load(model_id)
        self.prompt_encoder = self.prompt_encoder.to(self.device)
        
        
        self.test_dataset = CRSRecRLDataset(
        dataset=args.dataset, split='test',entity2id = self.kg['entity2id'], repeated_item_removed=args.repeated_item_removed, language = args.language, debug=args.debug,
        tokenizer=self.tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=self.text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length
    )
        self.data_collator = CRSRecRLDataCollator(
        tokenizer=self.tokenizer, device=self.device, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
        pad_entity_id=self.kg['pad_entity_id'],
        prompt_tokenizer=self.text_tokenizer, prompt_max_length=args.prompt_max_length,
    )

        self.dataloader = DataLoader(
        self.test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=self.data_collator,
    )
        self.evaluator_removed = RecMetricRemoved()
        self.evaluator_not_removed = RecMetricNotRemoved()
        
        
    @torch.no_grad()
    def evaluate(self, logger):
        self.model.eval()
        self.prompt_encoder.eval()
        self.text_encoder.eval()
        reports = {}
        # evaluate with repeated item removed
        for batch in tqdm(self.dataloader):
            raw_context = batch.pop('raw_context')
            index = batch.pop('index')
            entities = batch.pop('mentioned_entities')
            # print(batch)
            labels = batch['context'].pop('rec_labels')
            with torch.no_grad():
                token_embeds = self.text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds = self.prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True
            )
            batch['context']['prompt_embeds'] = prompt_embeds
            batch['context']['entity_embeds'] = self.prompt_encoder.get_entity_embeds()
            outputs = self.model(**batch['context'], rec=True)
            logits = outputs.rec_logits[:, self.item_ids]
            ranks = torch.topk(logits, k=100, dim=-1).indices
            preds = self.item_ids[ranks].tolist()
            self.evaluator_not_removed.evaluate(preds, labels, entities)
            preds_filtered = remove_repeated_items_batch(preds, entities)     
            self.evaluator_removed.evaluate(preds_filtered, labels, entities)
        
        reports['offline_metrics_removed'] = self.evaluator_removed.report()
        reports['offline_metrics_not_removed'] = self.evaluator_not_removed.report()
        for k, v in reports.items():
            logger.info(f'{k}: {v}')
        return reports
        
        
if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--log_dir", type=str, default='no.log')
        parser.add_argument("--debug", action='store_true', help="Debug mode.")
        parser.add_argument("--language", type=str, default='english')
        # data
        parser.add_argument("--dataset", type=str, default='redial_rec')
        parser.add_argument("--use_resp", default=False)
        parser.add_argument("--context_max_length", type=int, default=200)
        parser.add_argument("--prompt_max_length", type=int, default=200)
        parser.add_argument("--entity_max_length", type=int, default=43)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument("--tokenizer", type=str, default='microsoft/DialoGPT-small') #/data/wxh/model/dialogue_gpt
        parser.add_argument("--text_tokenizer", type=str, default='roberta-base') #/data/wxh/model/roberta-base
        # model
        parser.add_argument("--model", type=str, default='microsoft/DialoGPT-small') #/data/wxh/model/GPT2_CN
        parser.add_argument("--text_encoder", type=str, default='roberta-base') #/data/wxh/model/RoBERTa_cn
        parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
        parser.add_argument("--n_prefix_rec", type=int, default=10)
        parser.add_argument("--prompt_encoder", type=str, default='/home/wxh/UniCRS/src/redial_rec/best')#/data/UniCRS/redial_rec_unicrs_greedy_data_aug_pre_50_run_3/best
        # optim
        parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
        parser.add_argument('--fp16', action='store_true')
        parser.add_argument('--repeated_item_removed', action='store_true')
        args = parser.parse_args()
        return args
    args = parse_args()
    config = vars(args)
    final_evaluator = FinalEvaluator(args.prompt_encoder, args)
    logger.remove()
    logger.add(sys.stderr, level='DEBUG')
    logger.add(args.log_dir)
    logger.info(config)
    report = final_evaluator.evaluate(logger)
    print(report)

