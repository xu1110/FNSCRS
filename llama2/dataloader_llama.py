from collections import defaultdict

import os
import torch


class LLaMaDataCollatorForRec:
    def __init__(
        self, tokenizer, mode='multi_turn', device=torch.device('cpu'), debug=False, use_amp=False,
        context_max_length=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.mode = mode
        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length
        self.len_pre = 0
        self.len_post = 0
        if self.mode == 'instruction':
            self.pre = "<s>[INST] Pretend you are a movie recommender system. I will give you a conversation between a user and you (a recommender system). Based on the conversation, \
please reply me with recommended movies \
. Here is the conversation: { "
            self.pre = tokenizer.encode(self.pre, add_special_tokens=False)
            self.post = " } [/INST] "
            self.post = tokenizer.encode(self.post, add_special_tokens=False)

            
            self.len_pre = len(self.pre)
            self.len_post = len(self.post)
            
    def __call__(self, data_batch):
        input_batch = defaultdict(list)
        label_batch = []
        weight_batch = []

        for data in data_batch:
            if self.mode == 'instruction':
                pre = self.pre.copy()
                post = self.post.copy()
                pre.extend(data['context'])
                pre.extend(post)
                input_batch['input_ids'].append(pre)
            elif self.mode == 'multi_turn':
                input_batch['input_ids'].append(data['context'])
            label_batch.append(data['rec'])
            if 'weight' not in data:
                weight_batch.append(1)
            else:
                weight_batch.append(data['weight'])

        batch = {'weight': torch.as_tensor(weight_batch, device=self.device)}

        input_batch = self.tokenizer.pad(
            input_batch, max_length=self.context_max_length+self.len_pre+self.len_post,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
        )
        # print(input_batch['input_ids'])
        # a = self.tokenizer.batch_decode(input_batch['input_ids'])
        # print(a)
        input_batch['labels'] = label_batch

        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)

        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids

        batch['input'] = input_batch

        return batch

