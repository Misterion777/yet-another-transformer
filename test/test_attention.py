import torch

from src.attention import *
from src.modules import construct_future_mask

d_model = 64
h = 4
d_k = d_v = int(d_model / h)  # 16
b_size = 8

seq_len = 128


def test_attention():
    q = torch.randn(b_size, seq_len, d_k)
    k = torch.randn(b_size, seq_len, d_k)
    v = torch.randn(b_size, seq_len, d_v)

    a = DotProductAttention(d_k)
    r = a(q, k, v)
    assert r.size() == v.size()


def test_attention_mask():
    q = torch.randn(b_size, seq_len, d_k)
    k = torch.randn(b_size, seq_len, d_k)
    v = torch.randn(b_size, seq_len, d_v)

    m = construct_future_mask(seq_len, b_size)

    a = DotProductAttention(d_k)
    r = a(q, k, v, m)
    assert r.size() == v.size()


def test_attention_head():
    q = torch.randn(b_size, seq_len, d_k)
    k = torch.randn(b_size, seq_len, d_k)
    v = torch.randn(b_size, seq_len, d_v)

    a = AttentionHead(d_v, d_k, d_model)
    r = a(q, k, v)
    assert r.size() == torch.Size((b_size, seq_len, d_model))


def test_attention_multihead():
    q = torch.randn(b_size, seq_len, d_k)
    k = torch.randn(b_size, seq_len, d_k)
    v = torch.randn(b_size, seq_len, d_v)

    a = MultiHeadAttention(d_v, d_k, d_model)
    r = a(q, k, v)
    assert r.size() == torch.Size((b_size, seq_len, d_model))


test_attention_mask()
