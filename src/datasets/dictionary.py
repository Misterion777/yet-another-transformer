import re

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
    PAD_TOKEN = "<pad>"

    def __init__(self, tokenizer_type="re"):
        self.word2idx = {}
        self.idx2word = []
        self.tokenizer = TOKENIZERS[tokenizer_type]
        self._add_spec_tokens()

    @property
    def bos_id(self):
        return self.word2idx[self.BOS_TOKEN]

    @property
    def eos_id(self):
        return self.word2idx[self.EOS_TOKEN]

    @property
    def unk_id(self):
        return self.word2idx[self.UNK_TOKEN]

    @property
    def pad_id(self):
        return self.word2idx[self.PAD_TOKEN]

    def tokens2id(self, tokens, add_unknown=False):
        if add_unknown:
            return self.add_tokens(tokens)
        else:
            ids = []
            for tok in tokens:
                tok = self._sub_spec_token(tok)
                tok_id = self.word2idx.get(tok, self.unk_id)
                ids.append(tok_id)
            return ids

    def ids2tokens(self, ids):
        return [self.idx2word[id] for id in ids]

    def add_token(self, word):
        word = word.lower()
        word = self._sub_spec_token(word)
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_tokens(self, tokens):
        return [self.add_token(word) for word in tokens]

    def tokenize(self, text):
        text = text.lower()
        tokens = self.tokenizer(text)
        return tokens

    def _add_spec_tokens(self):
        self.add_tokens(
            [self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN, self.PAD_TOKEN]
        )

    def _sub_spec_token(self, token):
        if token == "\n":
            return self.EOS_TOKEN
        return token

    def __len__(self):
        return len(self.idx2word)
