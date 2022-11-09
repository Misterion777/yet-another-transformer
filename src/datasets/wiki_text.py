import torch
from torch.utils.data import Dataset

from src.constants import SEQ_LEN
from .dictionary import Dictionary

# https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
class WikiText(Dataset):
    def __init__(self, tokens_path: str, dictionary: Dictionary,seq_len=SEQ_LEN,build_dict=False):
        self.tokens_path = tokens_path
        self.dictionary = dictionary
        self.seq_len = seq_len

        self.tokens = []
        self.token_ids = []
        print(f"Building dataset defined by path: '{tokens_path}'")
        with open(self.tokens_path, "r", encoding="utf8") as f:
            for line in f:
                line_tokens = self.dictionary.tokenize(line)
                line_ids = self.dictionary.tokens2id(line_tokens,add_unknown=build_dict)
                self.tokens.extend(line_tokens)
                self.token_ids.extend(line_ids)
        print(f"Building finished. Dataset length: {len(self)}")

    def __len__(self):
        return len(self.token_ids) // self.seq_len

    def __getitem__(self, index):
        seq_len = min(self.seq_len, len(self.token_ids) - 1 - index)
        seq_len -= 2 # BOS AND EOS

        start_idx = index * seq_len
        end_idx = (index + 1) * seq_len

        data = torch.tensor([self.dictionary.bos_id] + self.token_ids[start_idx:end_idx] + [self.dictionary.eos_id])
        target = torch.tensor([self.dictionary.bos_id] + self.token_ids[start_idx+1:end_idx+1] + [self.dictionary.eos_id])
        return data, target
