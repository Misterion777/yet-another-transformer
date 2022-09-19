import torch
from torch import nn

from src.constants import DICT_SIZE


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()
        self.emb_size = emb_size

    def forward(self, x):
        batch_size = x.size(0)
        enc = torch.zeros(
            batch_size, x.size(1), self.emb_size
        )  # B, sentence_length, emb_size

        i = (
            2
            * torch.arange(self.emb_size).expand(batch_size, x.size(1), -1)
            / self.emb_size
        )  # B, sentence_length, emb_size
        pos = torch.arange(x.size(1)).expand(
            batch_size, self.emb_size, -1
        )  # B, emb_size, sentence_length
        pos = pos.reshape(
            batch_size, x.size(1), self.emb_size
        )  # B, sentence_length, emb_size

        sin = torch.sin(pos / (10000**i))
        cos = torch.cos(pos / (10000**i))
        enc[:, :, ::2] = sin[:, :, ::2]
        enc[:, :, 1::2] = cos[:, :, 1::2]
        return enc


class WordEmbedding(nn.Module):
    def __init__(self, emb_size: int, dict_size: int = DICT_SIZE):
        super().__init__()
        self.emb = nn.Embedding(dict_size, emb_size)
        self.pe = PositionalEncoding(emb_size)

    def forward(self, x):
        x = self.emb(x)
        return x + self.pe(x)
