from typing import Optional

import torch
from torch import nn

from src.constants import ATTENTION_HEADS, DROPOUT_P, FF_DIM
from src.utils import construct_future_mask

from .attention import MultiHeadAttention


class FFN(nn.Module):
    def __init__(self, in_dim: int, model_dim: int, ffn_size: int = FF_DIM):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, ffn_size)
        self.lin2 = nn.Linear(ffn_size, model_dim)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        attn_heads: int = ATTENTION_HEADS,
        dropout_p: float = DROPOUT_P,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            in_dim, in_dim, model_dim, attn_heads
        )
        self.drop1 = nn.Dropout(dropout_p)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, model_dim)
        self.drop2 = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(
        self, input: torch.Tensor, input_mask: Optional[torch.Tensor] = None
    ):
        attn_result = self.self_attn(input, input, input, input_mask)
        attn_result = self.drop1(attn_result)
        attn_result = self.norm1(input + attn_result)

        ffn_result = self.ffn(attn_result)
        ffn_result = self.drop2(ffn_result)

        ffn_result = self.norm2(attn_result + ffn_result)
        return ffn_result


class DecoderLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        encoder_dim: int,
        model_dim: int,
        attn_heads: int = ATTENTION_HEADS,
        dropout_p: float = DROPOUT_P,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            in_dim, in_dim, model_dim, attn_heads
        )
        self.drop1 = nn.Dropout(dropout_p)
        self.norm1 = nn.LayerNorm(model_dim)
        self.cross_attn = MultiHeadAttention(
            encoder_dim, model_dim, model_dim, attn_heads
        )
        self.drop2 = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, model_dim)
        self.drop3 = nn.Dropout(dropout_p)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(
        self,
        input: torch.Tensor,
        encoder_output: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, emb_dim = input.size()
        self_mask = construct_future_mask(seq_len, batch_size)
        attn_result = self.self_attn(input, input, input, self_mask)
        attn_result = self.drop1(attn_result)
        attn_result = self.norm1(input + attn_result)

        enc_result = self.cross_attn(
            attn_result, encoder_output, encoder_output, input_mask
        )
        enc_result = self.drop2(enc_result)
        enc_result = self.norm2(attn_result + enc_result)

        ffn_result = self.ffn(enc_result)
        ffn_result = self.drop3(ffn_result)
        ffn_result = self.norm2(enc_result + ffn_result)
        return ffn_result
