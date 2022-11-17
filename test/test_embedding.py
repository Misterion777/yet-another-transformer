import torch

from src.constants import DICT_SIZE
from src.embedding import *

d_model = 64
h = 4
d_k = d_v = int(d_model / h)  # 16
b_size = 8

seq_len = 128


def test_pe():
    batched_indices = torch.randint(
        low=0, high=DICT_SIZE, size=(b_size, seq_len)
    )

    pe = PositionalEncoding(d_model)
    r = pe(batched_indices)

    assert r.size() == torch.Size((b_size, batched_indices.size(1), d_model))


def test_emb():
    batched_indices = torch.randint(
        low=0, high=DICT_SIZE, size=(b_size, seq_len)
    )

    emb = WordEmbedding(d_model)
    r = emb(batched_indices)

    assert r.size() == torch.Size((b_size, batched_indices.size(1), d_model))
