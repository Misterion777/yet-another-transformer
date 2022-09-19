import torch
from torch import nn
from src.attention import MultiHeadAttention
from src.constants import ATTENTION_HEADS


class FFN(nn.Module):
    def __init__(self, in_dim: int, model_dim: int):
        self.lin1 = nn.Linear(in_dim, model_dim)
        self.lin2 = nn.Linear(in_dim, model_dim)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, in_dim: int, model_dim: int, attn_heads: int = ATTENTION_HEADS
    ):
        self.self_attn = MultiHeadAttention(
            in_dim, in_dim, model_dim, attn_heads
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, input: torch.Tensor):
        attn_result = self.self_attn(input, input, input)
        attn_result = self.norm1(input + attn_result)

        ffn_result = self.ffn(attn_result)
        ffn_result = self.norm2(attn_result + ffn_result)
        return ffn_result


class DecoderLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        encoder_dim: int,
        model_dim: int,
        attn_heads: int = ATTENTION_HEADS,
    ):
        self.self_attn = MultiHeadAttention(
            in_dim, in_dim, model_dim, attn_heads
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.encoder_attn = MultiHeadAttention(
            encoder_dim, model_dim, model_dim, attn_heads
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, input: torch.Tensor, encoder_output: torch.Tensor):
        attn_result = self.self_attn(input, input, input)
        attn_result = self.norm1(input + attn_result)

        enc_result = self.encoder_attn(
            encoder_output, encoder_output, attn_result
        )
        enc_result = self.norm2(attn_result + enc_result)

        ffn_result = self.ffn(enc_result)
        ffn_result = self.norm2(enc_result + ffn_result)
        return ffn_result
