import torch
from torch import nn
from src.constants import DICT_SIZE, N_LAYERS
from src.embedding import WordEmbedding
from src.modules import DecoderLayer, EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self, emb_dim: int, model_dim: int, num_layers: int = N_LAYERS
    ):
        super().__init__()
        self.emb = WordEmbedding(emb_dim)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(emb_dim, model_dim) for _ in range(num_layers)]
        )

    def forward(self, input: torch.Tensor):
        input = self.emb(input)
        for enc in self.encoder_layers:
            input = enc(input)
        return input


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

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(emb_dim, enc_dim, model_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, encoder_output: torch.Tensor):
        input = self.emb(input)

        for dec in self.decoder_layers:
            input = dec(input, encoder_output)

        return input


class GeneratorTransformer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        model_dim: int,
        num_layers: int = N_LAYERS,
        dict_size: int = DICT_SIZE,
    ):
        super().__init__()

        self.encoder = Encoder(emb_dim, model_dim, num_layers)
        self.decoder = Decoder(emb_dim, model_dim, model_dim, num_layers)

        self.tokens_proj = nn.Linear(model_dim, dict_size)

    def forward(self, seq_in: torch.Tensor, seq_out: torch.Tensor):
        enc_out = self.encoder(seq_in)
        dec_out = self.decoder(seq_out, enc_out)
        tokens_logits = self.tokens_proj(dec_out)
        return torch.softmax(tokens_logits, 1)
