from torch.utils.data import Dataset
from utils import sample_data


class DatasetForRec(Dataset):
    def __init__(
        self, data_list, tokenizer, entity2id, repeated_item_removed, mode='multi_turn', debug=False, shot=1,
        context_max_length=None
    ):
        super(DatasetForRec, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length
        self.repeated_item_removed = repeated_item_removed
        self.entity2id = entity2id
        self.mode = mode
        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)

    def prepare_data(self, data_list):
        for data in data_list:
            if len(data['rec']) == 0:
                continue

            text_list = []
            
            context = ""
            if self.mode == 'multi_turn':
                for i in range(int((len(data['context'])-1)/2)):
                    utt_usr = data['context'][i*2]
                    utt_sys = data['context'][i*2+1]
                    if utt_usr == '':
                        utt_usr = 'Hello.'
                    context += f'<s>[INST] {utt_usr} [/INST] {utt_sys} </s>'
                utt_usr = data['context'][-1]
                if utt_usr == '':
                    utt_usr = 'Hello.'
                context += f'<s>[INST] {utt_usr} [/INST] '
                # context += 'I recommend '
                # print(context)
            elif self.mode == 'instruction':
                turn_idx = 0
                text = ''
                for utt in data['context']:
                    if utt != '':
                        if turn_idx % 2 == 0:
                            text += ' User: '
                        else:
                            text += ' System: '
                        text += utt
                    turn_idx += 1
                context = text
            # print(context)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length, add_special_tokens=False)
            # print(context_ids)
            if 0 in context_ids:
                raise RuntimeError('Do not use 0 as padding!')
            
            # print(self.entity2id)
            for rec in data['rec']:
                if rec in self.entity2id:
                    data_dict = {
                        'context': context_ids,
                        'entity': [self.entity2id[ent] for ent in data['entity'] if ent in self.entity2id],
                        'rec': self.entity2id[rec],
                        # 'template': data['template'],
                    }
                    if 'template' in data:
                        data_dict['template'] = data['template']
                    
                    if self.repeated_item_removed:#  == 'True':
                        if self.entity2id[rec] not in data_dict['entity']:
                            self.data_list.append(data_dict)
                    else:
                        self.data_list.append(data_dict)


    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)

