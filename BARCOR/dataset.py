from torch.utils.data import Dataset
from tqdm import tqdm
from utils import sample_data
import torch.nn.functional as F
import copy

class DatasetForRec(Dataset):
    def __init__(
        self, data_list,repeated_item_removed , tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None
    ):
        super(DatasetForRec, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id
        self.repeated_item_removed = repeated_item_removed
        self.data_list = []
        # data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)
        self.data_list = sample_data(self.data_list, shot=shot, debug=debug)

    def prepare_data(self, data_list):
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
            # context_ids = self.tokenizer.encode(context, truncation=True, max_length=10)
            # print(context_ids)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # {'item': {'input_ids': [25, 25], 'attention_mask': [1, 1]},
            # 'bart': 
            # {'input_ids': [0, 44518, 35, 12289, 6, 38, 101, 814, 4133, 4, 2, 36383, 35, 1832, 47, 101, 814, 4133, 116, 2, 44518, 35, 3216, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}
            # print(self.entity2id)
            # print(len(data['rec']))
            for rec in data['rec']:
                if rec in self.entity2id:
                    data_dict = {
                        'raw_context': context,
                        'context': context_ids,
                        'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                        'rec': self.entity2id[rec],
                        # 'template': data['template'],
                    }
                    if 'template' in data:
                        data_dict['template'] = data['template']
                    
                    if self.repeated_item_removed:#  == 'true' or self.repeated_item_removed == 'True':
                        
                        if self.entity2id[rec] not in data_dict['entity']:
                            self.data_list.append(data_dict)
                    else:
                        self.data_list.append(data_dict)


    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)

class DatasetForRecAug(Dataset):
    def __init__(
        self, tokenizer, movie_ids, entity2id, debug=False, shot=1,
        context_max_length=None,
    ):
        super(DatasetForRecAug, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.movie_ids = movie_ids
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id

        self.data_list = []
        # data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data()

    def prepare_data(self):
        for entity, index in self.entity2id.items():
            if index not in self.movie_ids and index != len(self.entity2id)-1:
                context_ids = self.tokenizer.encode(entity)# , truncation=True, max_length=self.context_max_length
                data_dict = {
                        'context': context_ids,
                        'entity': [],
                        'rec': index,
                        # 'template': data['template'],
                    }
                self.data_list.append(data_dict)

    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)


class DatasetForConv(Dataset):
    def __init__(
        self, data_list, tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None, resp_max_length=None
    ):
        super(DatasetForConv, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id

        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)

    def prepare_data(self, data_list):
        for data in data_list:
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
            # User: Hi how are you</s>System: Hi there, what genre of movie do you enjoy?</s>User: happy new year</s>System: Happy New Year to you as well.</s>User: Have you seen any good comedy movies lately</s>System: Did you see @Ted?</s>User: Yes! Loved it.</s>System: Ted (2012) and Ted 2 (2015) were both very good. do you like Seth Macfarlane?</s>User: Yes, I liked them both a lot. Yes, I do. His new show orville is great. Talladega Nights: The Ballad of Ricky Bobby (2006) is one of my favorite movies
            if turn_idx % 2 == 0:
                user_str = 'User: '
            else:
                user_str = 'System: '
            resp = user_str + data['mask_resp']
            resp_ids = self.tokenizer.encode(resp, truncation=True, max_length=self.resp_max_length)
            data_dict = {
                'context': context_ids,
                'resp': resp_ids,
            }
            self.data_list.append(data_dict)

    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)

class DatasetForRecBpr(Dataset):
    def __init__(
        self, data_list,repeated_item_removed , tokenizer, entity2id, min_gap, debug=False, shot=1,
        context_max_length=None
    ):
        super(DatasetForRecBpr, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.min_gap = min_gap
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id
        self.repeated_item_removed = repeated_item_removed
        self.data_list = []
        self.prepare_data(data_list)
        self.data_list = sample_data(self.data_list, shot=shot, debug=debug)

    def prepare_data(self, data_list):
        for data in tqdm(data_list):
            if len(data['pairs']) == 0:
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
            # context_ids = self.tokenizer.encode(context, truncation=True, max_length=10)
            # print(context_ids)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # {'item': {'input_ids': [25, 25], 'attention_mask': [1, 1]},
            # 'bart': 
            # {'input_ids': [0, 44518, 35, 12289, 6, 38, 101, 814, 4133, 4, 2, 36383, 35, 1832, 47, 101, 814, 4133, 116, 2, 44518, 35, 3216, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}
            # print(self.entity2id)
            pos = []
            neg = []
            gap = []
            for pair in data['pairs']:
                # print(pair['accept'])
                # print(data['entity'])
                if pair['accept'] in self.entity2id and pair['reject'] in self.entity2id:
                    if self.repeated_item_removed and (pair['accept'] in data['entity'] or pair['reject'] in data['entity']):
                        # print(pair['accept'])
                        # print(data['entity'])
                        continue
                    if pair['gap'] >= self.min_gap:
                        gap.append(pair['gap'])
                        pos.append(self.entity2id[pair['accept']])
                        neg.append(self.entity2id[pair['reject']])
            if len(gap) != 0:
                data_dict = {
                        'raw_context': context,
                        'context': context_ids,
                        'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                        'pos': pos,
                        'neg': neg,
                        'gap': gap,
                        # 'template': data['template'],
                    }
                self.data_list.append(data_dict)
                    
            # for pair in data['pairs']:
            #     if pair['accept'] in self.entity2id and pair['reject'] in self.entity2id:
            #         data_dict = {
            #             'raw_context': context,
            #             'context': context_ids,
            #             'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
            #             'pos': self.entity2id[pair['accept']],
            #             'neg': self.entity2id[pair['reject']],
            #             'gap': pair['gap'],
            #             # 'template': data['template'],
            #         }
            #         if 'template' in data:
            #             data_dict['template'] = data['template']
                    
            #         if self.repeated_item_removed:#  == 'true' or self.repeated_item_removed == 'True':
            #             if data_dict['pos'] not in data_dict['entity'] and data_dict['neg'] not in data_dict['entity']:
            #                 self.data_list.append(data_dict)
            #         else:
            #             self.data_list.append(data_dict)


    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)

class DatasetForRecBprCeFast(Dataset):
    def __init__(
        self, data_list,repeated_item_removed , tokenizer, entity2id, min_gap, debug=False, shot=1,
        context_max_length=None
    ):
        super(DatasetForRecBprCeFast, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id
        self.min_gap = min_gap
        self.repeated_item_removed = repeated_item_removed
        self.data_list = []
        self.prepare_data(data_list)
        self.data_list = sample_data(self.data_list, shot=shot, debug=debug)

    def prepare_data(self, data_list):
        for data in tqdm(data_list):
            if len(data['pairs']) == 0 and len(data['rec']) == 0:
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
            # context_ids = self.tokenizer.encode(context, truncation=True, max_length=10)
            # print(context_ids)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # {'item': {'input_ids': [25, 25], 'attention_mask': [1, 1]},
            # 'bart': 
            # {'input_ids': [0, 44518, 35, 12289, 6, 38, 101, 814, 4133, 4, 2, 36383, 35, 1832, 47, 101, 814, 4133, 116, 2, 44518, 35, 3216, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}
            # print(self.entity2id)
            pos = []
            neg = []
            gap = []
            rec = []
            for r in data['rec']:
                if self.repeated_item_removed:#  == 'true' or self.repeated_item_removed == 'True':
                    if r not in data['entity']:
                        rec.append(self.entity2id[r])
                else:
                    rec.append(self.entity2id[r])
            if 'pairs' in data:
                for pair in data['pairs']:
                    # print(pair['accept'])
                    # print(data['entity'])
                    if pair['accept'] in self.entity2id and pair['reject'] in self.entity2id:
                        if self.repeated_item_removed and (pair['accept'] in data['entity'] or pair['reject'] in data['entity']):
                            # print(pair['accept'])
                            # print(data['entity'])
                            continue
                        if pair['gap'] >= self.min_gap:
                            gap.append(pair['gap'])
                            pos.append(self.entity2id[pair['accept']])
                            neg.append(self.entity2id[pair['reject']])
            if len(rec) != 0:
                if 'pairs' in data and len(gap) == 0:
                    continue
                data_dict = {
                        'raw_context': context,
                        'context': context_ids,
                        'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                        'pos': pos,
                        'neg': neg,
                        'gap': gap,
                        'rec': rec
                        # 'template': data['template'],
                    }
                self.data_list.append(data_dict)


    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)

class DatasetForRecBprCe(Dataset):
    def __init__(
        self, data_list,repeated_item_removed , tokenizer, entity2id, min_gap, debug=False, shot=1,
        context_max_length=None
    ):
        super(DatasetForRecBprCe, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id
        self.min_gap = min_gap
        self.repeated_item_removed = repeated_item_removed
        self.data_list = []
        self.prepare_data(data_list)
        self.data_list = sample_data(self.data_list, shot=shot, debug=debug)

    def prepare_data(self, data_list):
        for data in tqdm(data_list):
            if len(data['pairs']) == 0 and len(data['rec']) == 0:
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
            # context_ids = self.tokenizer.encode(context, truncation=True, max_length=10)
            # print(context_ids)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # {'item': {'input_ids': [25, 25], 'attention_mask': [1, 1]},
            # 'bart': 
            # {'input_ids': [0, 44518, 35, 12289, 6, 38, 101, 814, 4133, 4, 2, 36383, 35, 1832, 47, 101, 814, 4133, 116, 2, 44518, 35, 3216, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}
            # print(self.entity2id)
            pos = []
            neg = []
            gap = []
            rec = []
            for r in data['rec']:
                if self.repeated_item_removed:#  == 'true' or self.repeated_item_removed == 'True':
                    if r not in data['entity']:
                        rec.append(self.entity2id[r])
                else:
                    rec.append(self.entity2id[r])
            for pair in data['pairs']:
                # print(pair['accept'])
                # print(data['entity'])
                if pair['accept'] in self.entity2id and pair['reject'] in self.entity2id:
                    if self.repeated_item_removed and (pair['accept'] in data['entity'] or pair['reject'] in data['entity']):
                        # print(pair['accept'])
                        # print(data['entity'])
                        continue
                    if pair['gap'] >= self.min_gap:
                        gap.append(pair['gap'])
                        pos.append(self.entity2id[pair['accept']])
                        neg.append(self.entity2id[pair['reject']])
            if len(gap) != 0 and len(rec) != 0:
                for r in rec:
                    data_dict = {
                            'raw_context': context,
                            'context': context_ids,
                            'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                            'pos': pos,
                            'neg': neg,
                            'gap': gap,
                            'labels': r
                            # 'template': data['template'],
                        }
                    self.data_list.append(data_dict)


    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)


class DatasetForRecBce(Dataset):
    def __init__(
        self, data_list,repeated_item_removed , tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None,
    ):
        super(DatasetForRecBce, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length
        self.entity2id = entity2id
        self.repeated_item_removed = repeated_item_removed
        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)

    def prepare_data(self, data_list):
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
            # context_ids = self.tokenizer.encode(context, truncation=True, max_length=10)
            # print(context_ids)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # {'item': {'input_ids': [25, 25], 'attention_mask': [1, 1]},
            # 'bart': 
            # {'input_ids': [0, 44518, 35, 12289, 6, 38, 101, 814, 4133, 4, 2, 36383, 35, 1832, 47, 101, 814, 4133, 116, 2, 44518, 35, 3216, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}
            # print(self.entity2id)
            data_dict = {
                'raw_context': context,
                'context': context_ids,
                'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                # 'rec': self.entity2id[rec],
                # 'template': data['template'],
            }
            if 'template' in data:
                data_dict['template'] = data['template']
            rec = [0] * (len(self.entity2id)-1)
            for r in data['rec']:
                if self.repeated_item_removed:#  == 'true' or self.repeated_item_removed == 'True':
                    if self.entity2id[r] not in data_dict['entity']:
                        rec[self.entity2id[r]] = 1
                else:
                    rec[self.entity2id[r]] = 1
            data_dict['rec'] = rec
            if rec != [0] * (len(self.entity2id)-1):
                self.data_list.append(data_dict)

    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)

class DatasetForRecBprBce(Dataset):
    def __init__(
        self, data_list,repeated_item_removed , tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None,
    ):
        super(DatasetForRecBce, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length
        self.entity2id = entity2id
        self.repeated_item_removed = repeated_item_removed
        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)

    def prepare_data(self, data_list):
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
            # context_ids = self.tokenizer.encode(context, truncation=True, max_length=10)
            # print(context_ids)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # {'item': {'input_ids': [25, 25], 'attention_mask': [1, 1]},
            # 'bart': 
            # {'input_ids': [0, 44518, 35, 12289, 6, 38, 101, 814, 4133, 4, 2, 36383, 35, 1832, 47, 101, 814, 4133, 116, 2, 44518, 35, 3216, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}
            # print(self.entity2id)
            data_dict = {
                'raw_context': context,
                'context': context_ids,
                'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                # 'rec': self.entity2id[rec],
                # 'template': data['template'],
            }
            if 'template' in data:
                data_dict['template'] = data['template']
            rec = [0] * (len(self.entity2id)-1)
            for r in data['rec']:
                if self.repeated_item_removed:#  == 'true' or self.repeated_item_removed == 'True':
                    if self.entity2id[r] not in data_dict['entity']:
                        rec[self.entity2id[r]] = 1
                else:
                    rec[self.entity2id[r]] = 1
            data_dict['rec'] = rec
            if len(rec) != 0:
                self.data_list.append(data_dict)

    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)


class DatasetForRecConstruct(Dataset):
    def __init__(
        self, data_list,repeated_item_removed , tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None
    ):
        super(DatasetForRecConstruct, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id
        self.repeated_item_removed = repeated_item_removed
        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)

    def prepare_data(self, data_list):
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
            # context_ids = self.tokenizer.encode(context, truncation=True, max_length=10)
            # print(context_ids)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # {'item': {'input_ids': [25, 25], 'attention_mask': [1, 1]},
            # 'bart': 
            # {'input_ids': [0, 44518, 35, 12289, 6, 38, 101, 814, 4133, 4, 2, 36383, 35, 1832, 47, 101, 814, 4133, 116, 2, 44518, 35, 3216, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}
            # print(self.entity2id)
            
            data_dict = {
                'raw_context': context,
                'context': context_ids,
                'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                'rec': [self.entity2id[ent] for ent in data['rec'] if ent in self.entity2id],
                'dislike': [self.entity2id[ent] for ent in data['dislike'] if ent in self.entity2id],
                # 'template': data['template'],
            }
            self.data_list.append(data_dict)


    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)


class DatasetForRecCL(Dataset):
    def __init__(
        self, data_list,repeated_item_removed , tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None
    ):
        super(DatasetForRecCL, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id
        self.repeated_item_removed = repeated_item_removed
        self.data_list = []
        # data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)
        self.data_list = sample_data(self.data_list, shot=shot, debug=debug)

    def remove_sublist_elements(self, a, b):
        a_set = set(a)  # 将列表 a 转换为集合
        # b_set = set(b)
        # result = list(b_set-a_set)
        result = [element for element in b if element not in a_set]  # 使用集合查找
        return result
    
    def prepare_data(self, data_list):
        for data in tqdm(data_list):
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
            # context_ids = self.tokenizer.encode(context, truncation=True, max_length=10)
            # print(context_ids)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # {'item': {'input_ids': [25, 25], 'attention_mask': [1, 1]},
            # 'bart': 
            # {'input_ids': [0, 44518, 35, 12289, 6, 38, 101, 814, 4133, 4, 2, 36383, 35, 1832, 47, 101, 814, 4133, 116, 2, 44518, 35, 3216, 2, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}
            # print(self.entity2id)
            
            if 'false_negatives' in data:
                fns = copy.deepcopy(data['false_negatives']).extend(copy.deepcopy(data['rec']))
                rns = data['real_negatives']
            else:
                fns = copy.deepcopy(data['rec'])
                if 'rec_original' in data:
                    fns.extend(copy.deepcopy(data['rec_original']))
                rns = []
            for rec in data['rec']:
                if rec in self.entity2id:
                    data_dict = {
                        'raw_context': context,
                        'context': context_ids,
                        'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                        'rec': self.entity2id[rec],
                        'fns': [self.entity2id[ent] for ent in fns if ent in self.entity2id],
                        'rns': [self.entity2id[ent] for ent in rns if ent in self.entity2id],
                        # 'template': data['template'],
                    }
                    # ns = self.remove_sublist_elements(data_dict['fns'], full_list)
                    data_dict['ns'] = []
                    if 'template' in data:
                        data_dict['template'] = data['template']
                    
                    if self.repeated_item_removed:#  == 'true' or self.repeated_item_removed == 'True':
                        
                        if self.entity2id[rec] not in data_dict['entity']:
                            self.data_list.append(data_dict)
                    else:
                        self.data_list.append(data_dict)


    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)
