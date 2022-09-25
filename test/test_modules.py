import torch

from src.modules import *

d_model = 64
h = 4
d_k = d_v = int(d_model / h)  # 16
b_size = 8

seq_len = 128


def test_ffn():
    input = torch.randn(b_size, seq_len, d_model)

    ffn = FFN(d_model, d_model)
    r = ffn(input)
    assert r.size() == torch.Size([b_size, seq_len, d_model])


def test_encoder_layer():
    input = torch.randn(b_size, seq_len, d_model)

    enc = EncoderLayer(d_model, d_model)
    r = enc(input)
    assert r.size() == torch.Size([b_size, seq_len, d_model])


def test_decoder_layer():
    input = torch.randn(b_size, seq_len, d_model)
    enc_output = torch.randn(b_size, seq_len, d_model)

    dec = DecoderLayer(d_model, d_model, d_model)
    r = dec(input, enc_output)
    assert r.size() == torch.Size([b_size, seq_len, d_model])
