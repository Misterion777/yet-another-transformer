import torch
from torch import nn
from src.constants import N_LAYERS
from src.embedding import WordEmbedding
from src.modules import DecoderLayer, EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self, emb_dim: int, model_dim: int, num_layers: int = N_LAYERS
    ):
        super().__init__()
        self.emb = WordEmbedding(emb_dim)

        layers = [EncoderLayer(emb_dim, model_dim) for _ in range(num_layers)]
        self.encoder_layers = nn.Sequential(*layers)  # Is it sequential?

    def forward(self, input):
        input = self.emb(input)
        return self.encoder_layers(input)


class Decoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        enc_dim: int,
        model_dim: int,
        num_layers: int = N_LAYERS,
    ):
        super().__init__()
        self.emb = WordEmbedding(emb_dim)

        layers = [
            DecoderLayer(emb_dim, enc_dim, model_dim)
            for _ in range(num_layers)
        ]
        self.decoder_layers = nn.Sequential(*layers)  # Is it sequential?

        self.final_proj = nn.Linear(model_dim, model_dim)

    def forward(self, input):
        input = self.emb(input)
        input = self.decoder_layers(input)
        input = self.final_proj(input)
        return torch.softmax(input, 1)
