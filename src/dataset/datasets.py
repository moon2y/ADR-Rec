import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd

# For reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Extract item index range
def count_unique_items(list_of_lists):
    all_elements = set()
    for inner_list in list_of_lists:
        for element in inner_list:
            all_elements.add(element)
    return min(all_elements), max(all_elements)

class BERTRecDataset(Dataset):
    def __init__(self, user_train, max_len, num_user, item_min, num_item, mask_prob=0.15):
        self.user_train = user_train
        self.max_len = max_len
        self.num_user = num_user
        self.item_min = item_min
        self.num_item = num_item
        self.mask_prob = mask_prob
        self._all_items = set(range(item_min, num_item + 1))

    def __len__(self):
        return self.num_user

    def __getitem__(self, user):
        user_seq = self.user_train[user]
        tokens = user_seq[-self.max_len:]
        padding = [0] * (self.max_len - len(tokens))
        tokens = padding + tokens
        return torch.LongTensor(tokens)

    def random_neg_sampling(self, rated_item, num_item_sample):
        return random.sample(list(self._all_items - set(rated_item)), num_item_sample)

class MakeSequenceDataSet:
    def __init__(self, data_path, cross_emb_path, spe_emb_path):
        self.users = pd.read_pickle(data_path)
        self.cross = np.vstack(pd.read_pickle(cross_emb_path))
        self.spe = np.vstack(pd.read_pickle(spe_emb_path))
        self.user_train, self.user_label, self.user_valid = {}, {}, {}

        for user in self.users:
            self.user_train[user] = self.users[user][-12:-2]
            self.user_label[user] = self.users[user][-2]
            self.user_valid[user] = self.users[user][-1]

        self.num_user = len(self.user_train)

    def get_train_valid_data(self):
        return self.user_train, self.user_label, self.user_valid, self.cross, self.spe