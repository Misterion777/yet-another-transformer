import torch

from src.constants import DICT_SIZE
from src.model.transformer import *

d_model = 64
h = 4
d_k = d_v = int(d_model / h)  # 16
b_size = 8

seq_len = 128


def test_encoder():
    input = torch.randint(low=0, high=DICT_SIZE, size=(b_size, seq_len))
    enc = Encoder(d_model, d_model)

    r = enc(input)
    assert r.size() == torch.Size([b_size, seq_len, d_model])


def test_decoder():
    input = torch.randint(low=0, high=DICT_SIZE, size=(b_size, seq_len))
    enc_output = torch.randn(b_size, seq_len, d_model)

    dec = Decoder(d_model, d_model, d_model)
    r = dec(input, enc_output)
    assert r.size() == torch.Size([b_size, seq_len, d_model])


def test_generator():
    seq1 = torch.randint(low=0, high=DICT_SIZE, size=(b_size, seq_len))

    seq2 = torch.randint(low=0, high=DICT_SIZE, size=(b_size, seq_len))

    model = GeneratorTransformer(d_model, d_model)

    r = model(seq1, seq2)

    assert r.size() == torch.Size([b_size, seq_len, DICT_SIZE])
