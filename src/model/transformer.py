from typing import Optional

import torch
from torch import nn

from src.constants import DICT_SIZE, N_LAYERS

from .embedding import WordEmbedding
from .modules import DecoderLayer, EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        model_dim: int,
        num_layers: int = N_LAYERS,
        dict_size: int = DICT_SIZE,
    ):
        super().__init__()
        self.emb = WordEmbedding(emb_dim, dict_size)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(emb_dim, model_dim) for _ in range(num_layers)]
        )

    def forward(
        self, input: torch.Tensor, input_mask: Optional[torch.Tensor] = None
    ):
        input = self.emb(input)
        for enc in self.encoder_layers:
            input = enc(input, input_mask)
        return input


class Decoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        enc_dim: int,
        model_dim: int,
        num_layers: int = N_LAYERS,
        dict_size: int = DICT_SIZE,
    ):
        super().__init__()
        self.emb = WordEmbedding(emb_dim, dict_size)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(emb_dim, enc_dim, model_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_output: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ):
        input = self.emb(input)

        for dec in self.decoder_layers:
            input = dec(input, encoder_output, input_mask)

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

        self.encoder = Encoder(emb_dim, model_dim, num_layers, dict_size)
        self.decoder = Decoder(
            emb_dim, model_dim, model_dim, num_layers, dict_size
        )

        self.tokens_proj = nn.Linear(model_dim, dict_size)

    def forward(
        self,
        seq_in: torch.Tensor,
        seq_out: torch.Tensor,
        seq_in_mask: Optional[torch.Tensor] = None,
    ):
        enc_out = self.encoder(seq_in, seq_in_mask)
        dec_out = self.decoder(seq_out, enc_out, seq_in_mask)
        tokens_logits = self.tokens_proj(dec_out)
        return torch.softmax(tokens_logits, 1)

    @torch.no_grad()
    def generate(
        self, seq_in: torch.Tensor, bos_id: int, max_length: int = 16
    ):
        batch_size = seq_in.size(0)
        enc_out = self.encoder(seq_in)
        dec_in = torch.full(
            (
                batch_size,
                1,
            ),
            bos_id,
            device=seq_in.device,
        )
        for i in range(max_length):
            dec_out = self.decoder(dec_in, enc_out)
            tokens_logits = self.tokens_proj(dec_out)
            # Take the argmax over the softmax of the last token to obtain the next-token prediction
            predicted_tokens = torch.argmax(
                tokens_logits[:, -1, :], dim=-1
            ).unsqueeze(1)

            # Append the prediction to the already decoded tokens and construct the new mask
            dec_in = torch.cat((dec_in, predicted_tokens), dim=-1)
        return dec_in[:, 1:]  # ignore bos token

    def load_checkpoint(self, load_path: str, device: torch.device):
        loaded = torch.load(load_path, map_location=device)
        self.load_state_dict(loaded)
