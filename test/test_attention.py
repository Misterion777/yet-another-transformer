import torch

from src.attention import *

d_model = 64
h = 4
d_k = d_v = int(d_model / h)  # 16
b_size = 8


def test_attention():
    q = torch.randn(b_size, d_k)
    k = torch.randn(b_size, d_k)
    v = torch.randn(b_size, d_v)

    a = DotProductAttention(d_k)
    r = a(q, k, v)
    assert r.size() == v.size()


def test_attention_head():
    q = torch.randn(b_size, d_k)
    k = torch.randn(b_size, d_k)
    v = torch.randn(b_size, d_v)

    a = AttentionHead(d_v, d_k, d_model)
    r = a(q, k, v)
    assert r.size() == torch.Size((b_size, d_model))


def test_attention_multihead():
    q = torch.randn(b_size, d_k)
    k = torch.randn(b_size, d_k)
    v = torch.randn(b_size, d_v)

    a = MultiHeadAttention(d_v, d_k, d_model)
    r = a(q, k, v)
    assert r.size() == torch.Size((b_size, d_model))
