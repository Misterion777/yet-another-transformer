import re
from typing import Literal

import torch
from torch.utils.data import Dataset

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
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, tokenizer_type: Literal["re", "space", "split"] = "re"):
        self.word2idx = {}
        self.idx2word = []
        self.tokenizer = TOKENIZERS[tokenizer_type]
        self._add_spec_tokens()

    def add_token(self, word):
        if word == "\n":
            word = self.EOS_TOKEN
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
        self.add_token(self.EOS_TOKEN)
        self.add_token(self.UNK_TOKEN)

    def __len__(self):
        return len(self.idx2word)


# https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
class WikiText(Dataset):
    def __init__(self, tokens_path: str, dictionary: Dictionary):
        self.tokens_path = tokens_path
        self.dictionary = dictionary

        self.tokens = []
        self.token_ids = []
        with open(self.tokens_path, "r", encoding="utf8") as f:
            for line in f:
                line_tokens = self.dictionary.tokenize(line)
                line_ids = self.dictionary.add_tokens(line_tokens)
                self.tokens.extend(line_tokens)
                self.token_ids.extend(line_ids)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, index):
        return torch.tensor(self.token_ids[index])


if __name__ == "__main__":
    wiki_dict = Dictionary()
    # train_ds = WikiText("data/wikitext-2/wiki.train.tokens",wiki_dict)
    val_ds = WikiText("data/wikitext-2/wiki.valid.tokens", wiki_dict)
    print(len(wiki_dict))
    # print(len(train_ds))
    print(len(val_ds))
