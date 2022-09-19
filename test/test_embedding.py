import torch

from src.embedding import *
from src.constants import DICT_SIZE

d_model = 64
h = 4
d_k = d_v = int(d_model / h)  # 16
b_size = 8

sent_len = 128


def test_pe():
    batched_indices = torch.randint(
        low=0, high=DICT_SIZE, size=(b_size, sent_len)
    )

    pe = PositionalEncoding(d_model)
    r = pe(batched_indices)

    assert r.size() == torch.Size((b_size, batched_indices.size(1), d_model))


def test_emb():
    batched_indices = torch.randint(
        low=0, high=DICT_SIZE, size=(b_size, sent_len)
    )

    emb = WordEmbedding(d_model)
    r = emb(batched_indices)

    assert r.size() == torch.Size((b_size, batched_indices.size(1), d_model))


# q = torch.randn(b_size, d_k)
# k = torch.randn(b_size, d_k)
# v = torch.randn(b_size, d_v)

# a = AttentionHead(d_v,d_k,d_model)
# r = a(q, k, v)
# print(r)
# print(r.size())
