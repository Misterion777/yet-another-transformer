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

    def forward(self, x):
        x = self.emb(x)
        return self.encoder_layers(x)


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

    def forward(self, x):
        x = self.emb(x)
        x = self.decoder_layers(x)
        x = self.final_proj(x)
        return torch.softmax(x, 1)
