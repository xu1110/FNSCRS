from typing import List, Union, Optional
import json
import torch
import random

def sample_data(data_list, shot=1, debug=False, number_for_debug=4096):
    if debug:
        data_list = data_list[:number_for_debug]

    if shot < 1:
        data_idx = random.sample(range(len(data_list)), int(len(data_list) * shot))
        data_list = [data_list[idx] for idx in data_idx]
    elif shot > 1:
        data_idx = random.sample(range(len(data_list)), int(shot))
        data_list = [data_list[idx] for idx in data_idx]

    return data_list

def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False
) -> torch.LongTensor:
    
    
    """Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(lens)
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)
    if debug and max_len is not None:
        t = max(t, max_len)

    if use_amp:
        t = t // 8 * 8

    output = torch.full((n, t), fill_value=pad_idx, dtype=torch.long, device=device)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length:] = item

    return output


class entity_database():
    def __init__(self,entity2id_path):
        self.entity2id = json.load(open(entity2id_path+'entity2id.json', 'r', encoding='utf-8'))
        self.movie_id = json.load(open(entity2id_path+'movie_ids.json', 'r', encoding='utf-8'))
        self.relation2id = json.load(open(entity2id_path+'relation2id.json', 'r', encoding='utf-8'))
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        
    def get_entity_name(self,id):
        return self.id2entity[id]
    
    def get_entity_id(self,name):
        return self.entity2id[name]
    
if __name__ == '__main__':
    entity_processor = entity_database('/home/lvchangze/xhz_code/UniCRS/src/data/redial/')
    print(len(entity_processor.movie_id))
    
        