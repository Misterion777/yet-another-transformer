import torch
from torch.utils.data import Dataset

from src.constants import SEQ_LEN
from .dictionary import Dictionary

# Processed from: https://www.kaggle.com/datasets/blessondensil294/friends-tv-series-screenplay-script?resource=download
class FriendsDialog(Dataset):
    def __init__(self, txt_path: str, dictionary: Dictionary,seq_len=SEQ_LEN, build_dict=False):
        self.txt_path = txt_path
        self.dictionary = dictionary
        self.seq_len = seq_len

        self.token_pairs = []
        self.token_ids_pairs = []
        print(f"Building dataset defined by path: '{txt_path}'")
        with open(self.txt_path, "r", encoding="utf8") as f:
            for line in f:
                ut1, ut2 = line.split("|")
                ut1_tokens = self.dictionary.tokenize(ut1)
                ut1_ids = self.dictionary.tokens2id(ut1_tokens,add_unknown=build_dict)

                ut2_tokens = self.dictionary.tokenize(ut2)
                ut2_ids = self.dictionary.tokens2id(ut2_tokens,add_unknown=build_dict)
                self.token_pairs.append((ut1_tokens,ut2_tokens))
                self.token_ids_pairs.append((ut1_ids,ut2_ids))
        print(f"Building finished. Dataset length: {len(self)}")

    def __len__(self):
        return len(self.token_ids_pairs)

    def __getitem__(self, index):
        ut1, ut2 = self.token_ids_pairs[index]
        ut1 = ut1[:self.seq_len - 2]
        ut2 = ut2[:self.seq_len - 2]

        ut1_pad_len = max(self.seq_len - len(ut1) - 2,0)
        ut2_pad_len = max(self.seq_len - len(ut2) - 2,0)

        data = torch.tensor([self.dictionary.bos_id] + ut1 + [self.dictionary.eos_id] + [self.dictionary.pad_id] * ut1_pad_len)
        target = torch.tensor([self.dictionary.bos_id] + ut2 + [self.dictionary.eos_id] + [self.dictionary.pad_id] * ut2_pad_len)        
        return data, target