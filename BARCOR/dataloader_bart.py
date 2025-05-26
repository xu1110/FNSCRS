from collections import defaultdict

import os
import torch


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
        fns_batch = []
        rns_batch = []
        ns_batch = []
        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            if 'raw_context' in data.keys():
                raw_context.append(data['raw_context'])
            label_batch.append(data['rec'])
            if 'fns' in data:
                fns_batch.append(data['fns'])
                rns_batch.append(data['rns'])
                ns_batch.append(data['ns'])
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
        
        
        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)
        if 'fns' in data:
            input_batch['fns'] = fns_batch
            input_batch['rns'] = rns_batch
            input_batch['labels'] = label_batch
            input_batch['ns'] = ns_batch
        else:
            input_batch['labels'] = torch.as_tensor(label_batch, device=self.device)
        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids

        batch['input'] = input_batch
        
        # print(batch)
        
        if mode == 'infer':
            return batch, raw_context
        else:
            return batch

class BARTDataCollatorForRecCL:
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
        fns_batch = []
        rns_batch = []
        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            if 'raw_context' in data.keys():
                raw_context.append(data['raw_context'])
            label_batch.append(data['rec'])
            fns_batch.append(data['fns'])
            rns_batch.append(data['rns'])
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
        
        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['fns'] = fns_batch
        input_batch['rns'] = rns_batch
        
        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids

        batch['input'] = input_batch
        
        # print(batch)
        
        if mode == 'infer':
            return batch, raw_context
        else:
            return batch


class BARTDataCollatorForConv:
    def __init__(
        self, tokenizer, device=torch.device('cpu'), debug=False, use_amp=False,
        context_max_length=None, resp_max_length=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length

    def __call__(self, data_batch):
        input_batch = defaultdict(list)
        label_batch = defaultdict(list)

        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            label_batch['input_ids'].append(data['resp'])

        input_batch = self.tokenizer.pad(
            input_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )

        label_ids = self.tokenizer.pad(
            label_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )['input_ids']
        input_batch['labels'] = label_ids

        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)

        return input_batch

class BARTDataCollatorForRecBpr:
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
        pos_batch = []
        neg_batch = []
        gap_batch = []
        weight_batch = []
        raw_context = []
        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            if 'raw_context' in data.keys():
                raw_context.append(data['raw_context'])
            pos_batch.append(data['pos'])
            neg_batch.append(data['neg'])
            gap_batch.append(data['gap'])
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
        
        
        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['pos'] = pos_batch
        input_batch['neg'] = neg_batch
        input_batch['gap'] = gap_batch
        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids

        batch['input'] = input_batch
        
        # print(batch)
        
        if mode == 'infer':
            return batch, raw_context
        else:
            return batch

class BARTDataCollatorForBce:
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
        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            if 'raw_context' in data.keys():
                raw_context.append(data['raw_context'])
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
        
        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                if k == 'labels':
                    input_batch[k] = torch.as_tensor(v, device=self.device, dtype=torch.float)
                else:
                    input_batch[k] = torch.as_tensor(v, device=self.device)
        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids
        batch['input'] = input_batch
        
        # print(batch)
        
        if mode == 'infer':
            return batch, raw_context
        else:
            return batch

class BARTDataCollatorForRecCeBpr:
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
        pos_batch = []
        neg_batch = []
        gap_batch = []
        label_batch = []
        weight_batch = []
        raw_context = []
        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            if 'raw_context' in data.keys():
                raw_context.append(data['raw_context'])
            pos_batch.append(data['pos'])
            neg_batch.append(data['neg'])
            gap_batch.append(data['gap'])
            label_batch.append(data['labels'])
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
        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['pos'] = pos_batch
        input_batch['neg'] = neg_batch
        input_batch['gap'] = gap_batch
        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids

        batch['input'] = input_batch
        
        # print(batch)
        
        if mode == 'infer':
            return batch, raw_context
        else:
            return batch

class BARTDataCollatorForRecCeBprFast:
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
        pos_batch = []
        neg_batch = []
        gap_batch = []
        label_batch = []
        weight_batch = []
        raw_context = []
        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            if 'raw_context' in data.keys():
                raw_context.append(data['raw_context'])
            pos_batch.append(data['pos'])
            neg_batch.append(data['neg'])
            gap_batch.append(data['gap'])
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
        
        
        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['pos'] = pos_batch
        input_batch['neg'] = neg_batch
        input_batch['gap'] = gap_batch
        input_batch['rec'] = label_batch
        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids

        batch['input'] = input_batch
        
        # print(batch)
        
        if mode == 'infer':
            return batch, raw_context
        else:
            return batch


class BARTDataCollatorForRecConstruct:
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
        dislike_batch = []
        weight_batch = []
        raw_context = []
        mentioned_entities = []
        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            if 'raw_context' in data.keys():
                raw_context.append(data['raw_context'])
            label_batch.append(data['rec'])
            dislike_batch.append(data['dislike'])
            mentioned_entities.append(data['entity'])
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
        
        
        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['labels'] = label_batch
        input_batch['dislikes'] = dislike_batch
        input_batch['entity'] = mentioned_entities
        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids
        batch['input'] = input_batch
        
        # print(batch)
        
        if mode == 'infer':
            return batch, raw_context
        else:
            return batch
