import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils import padded_tensor, sample_data


class CRSRecDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, entity2id, repeated_item_removed, language, debug=False,
        context_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,
        use_resp=False, path = None, max_rec = None, shot = 1
    ):
        super(CRSRecDataset, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.repeated_item_removed = repeated_item_removed
        self.prompt_tokenizer = prompt_tokenizer
        self.use_resp = use_resp
        self.entity2id = entity2id
        self.context_max_length = context_max_length
        self.language = language
        self.shot = shot
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1
        self.path = path
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length
        self.max_rec = max_rec if max_rec != None else 10000
        dataset_dir = os.path.join('data', dataset)
        if path == None:
            data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        else:
            data_file = os.path.join(dataset_dir, f'{path}')
        self.data = []
        self.prepare_data(data_file)
        self.data = sample_data(self.data, shot = self.shot)
        
    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]

            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:
                    continue
                if len(dialog['context']) == 1 and dialog['context'][0] == '':
                    continue

                context = ''
                prompt_context = ''
                # already_mentioned = []
                
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
                # print(context)
                # print(prompt_context)
                # print(context_ids)
                # print(prompt_ids)
                # print("\n")
                if len(dialog['rec']) != 0 and len(dialog['entity']) != 0:
                    for item in dialog['rec'][:self.max_rec]:
                        if item in self.entity2id:
                            data = {
                                'context': context_ids,
                                'entity': [self.entity2id[entityid] for entityid in dialog['entity'][-self.entity_max_length:] if entityid in self.entity2id],
                                'rec': self.entity2id[item],
                                'prompt': prompt_ids,
                                'raw_context': context
                            }
                            if self.repeated_item_removed:# == 'True' or self.repeated_item_removed == 'true':
                                if data['rec'] not in data['entity']:
                                    self.data.append(data)
                            else:
                                self.data.append(data)

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class CRSRecDataCollator:
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
        for data in data_batch:
            # input_ids = data['context'][-(self.context_max_length - len(self.rec_prompt_ids)):] + self.rec_prompt_ids
            input_ids = data['context']
            context_batch['input_ids'].append(input_ids)
            entity_batch.append(data['entity'])
            label_batch.append(data['rec'])
            prompt_batch['input_ids'].append(data['prompt'])
            raw_context = data['raw_context']
            raw_context_batch.append(raw_context)
        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length
        )
        context_batch['rec_labels'] = label_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
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

        entity_batch = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
        input_batch['entity'] = entity_batch
        input_batch['raw_context']=raw_context_batch
        return input_batch

