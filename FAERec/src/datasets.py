
import torch
from utils import neg_sample,get_sample_negs
from torch.utils.data import Dataset


class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors
    
    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        
        assert self.data_type in {"train", "valid", "test"}

        if len(items)<=3:
            input_ids,target_pos,answer=filter_user(items,self.data_type)

        else:
            if self.data_type == "train":
                input_ids = items[:-3]
                target_pos = items[1:-2]
                answer = [0]  

            elif self.data_type == "valid":
                input_ids = items[:-2]
                target_pos = items[1:-1]
                answer = [items[-2]]

            else:
                input_ids = items[:-1]
                target_pos = items[1:]
                answer = [items[-1]]
                
        return self._data_sample_rec_task(user_id, items, input_ids,target_pos,answer)

    def __len__(self):
        return len(self.user_seq)

# Data processing, all baselines remain consistent.
def filter_user(items, data_type):
    seq_len = len(items)
    
    if seq_len == 1:
        return [], [], [0]

    elif seq_len == 2:
        if data_type == "train":
            
            input_ids = [items[0]]
            target_pos = [items[1]]
            answer = [0]
            return input_ids, target_pos, answer
        else:
            return [], [], [0]

    elif seq_len == 3:
        if data_type == "train":
            return [], [], [0]
        elif data_type == "valid":
            input_ids = [items[0]]
            target_pos = [0] 
            answer = [items[1]]
            return input_ids, target_pos, answer
        else:  # test
            input_ids = [items[0], items[1]]
            target_pos = [0,0]  
            answer = [items[2]]
            return input_ids, target_pos, answer
    