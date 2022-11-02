import re
from typing import Literal

import torch
from torch.utils.data import Dataset, DataLoader

from src.constants import SEQ_LEN,BATCH_SIZE

RETOK = re.compile(r"\w+|[^\w\s]|\n", re.UNICODE)


def re_tokenize(text):
    r"""
    Tokenize using a liberal regular expression.
    Find boundaries between word characters, newlines, and non-word
    non-whitespace tokens ``(r'[\\w\\n]+ | [^\\w\\s] | \\n')``.
    This splits along whitespace and punctuation and keeps the newline as
    a token in the returned list.
    """
    return RETOK.findall(text)


def split_tokenize(text):
    """
    Tokenize on whitespace and some limited punctuation.
    Splits tokens based on whitespace after adding whitespace around
    punctuation.
    Use re_tokenize if you want more robust handling of punctuation.
    """
    return (
        text.replace(".", " . ")
        .replace(",", " , ")
        .replace(";", " ; ")
        .replace(":", " : ")
        .replace("!", " ! ")
        .replace("?", " ? ")
        .split()
    )


def space_tokenize(text):
    """
    Tokenize exactly on spaces.
    Useful when text is pre-tokenized.
    """
    return text.strip().split(" ")


TOKENIZERS = {
    "re": re_tokenize,
    "space": space_tokenize,
    "split": split_tokenize,
}

class Dictionary:
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, tokenizer_type: Literal["re", "space", "split"] = "re"):
        self.word2idx = {}
        self.idx2word = []
        self.tokenizer = TOKENIZERS[tokenizer_type]
        self._add_spec_tokens()

    def tokens2id(self,tokens,add_unknown=False):
        if add_unknown:
            return self.add_tokens(tokens)
        else:
            ids = []
            for tok in tokens:
                tok = self._sub_spec_token(tok)                
                tok_id = self.word2idx.get(tok,self.word2idx[self.UNK_TOKEN])                
                ids.append(tok_id)
            return ids

    def add_token(self, word):
        word = self._sub_spec_token(word)
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_tokens(self, tokens):
        return [self.add_token(word) for word in tokens]

    def tokenize(self, text):
        tokens = self.tokenizer(text)
        return tokens

    def _add_spec_tokens(self):
        self.add_tokens(self.BOS_TOKEN,self.EOS_TOKEN,self.UNK_TOKEN)

    def _sub_spec_token(self,token):
        if token == "\n":
            return self.EOS_TOKEN
        return token

    def __len__(self):
        return len(self.idx2word)


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

        start_idx = index * seq_len
        end_idx = (index + 1) * seq_len

        data = torch.tensor(self.token_ids[start_idx:end_idx])
        target = torch.tensor(self.token_ids[start_idx+1:end_idx+1])
        return data, target        


if __name__ == "__main__":
    wiki_dict = Dictionary()
    train_ds = WikiText("data/wikitext-2/wiki.train.tokens",wiki_dict,build_dict=True)
    val_ds = WikiText("data/wikitext-2/wiki.valid.tokens", wiki_dict)
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)
    print(len(wiki_dict))
    # print(len(train_ds))
    print(len(val_ds))
