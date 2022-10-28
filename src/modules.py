import torch
from torch import nn
from src.attention import MultiHeadAttention
from src.constants import ATTENTION_HEADS


class FFN(nn.Module):
    def __init__(self, in_dim: int, model_dim: int):
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.self_attn = MultiHeadAttention(
            in_dim, in_dim, model_dim, attn_heads
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.cross_attn = MultiHeadAttention(
            encoder_dim, model_dim, model_dim, attn_heads
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, input: torch.Tensor, encoder_output: torch.Tensor):
        batch_size, seq_len, emb_dim = input.size()
        self_mask = construct_future_mask(seq_len, batch_size)
        attn_result = self.self_attn(input, input, input, self_mask)
        attn_result = self.norm1(input + attn_result)

        enc_result = self.cross_attn(
            attn_result, encoder_output, encoder_output
        )
        enc_result = self.norm2(attn_result + enc_result)

        ffn_result = self.ffn(enc_result)
        ffn_result = self.norm2(enc_result + ffn_result)
        return ffn_result


# Idea is taken from https://github.com/jsbaan/transformer-from-scratch/blob/main/utils.py
def construct_future_mask(seq_len: int, batch_size: int = 1):
    """
    Construct a binary mask that contains 1's for all previous connections (autoregressive) and 0's for all outgoing future connections.
    This mask will be applied to the attention logits in decoder self-attention such that all logits with a 0 mask are set to -inf.
    :param seq_len: length of the input sequence
    :return: (seq_len,seq_len) mask
    """
    subsequent_mask = torch.tril(
        torch.ones(batch_size, seq_len, seq_len), diagonal=-1
    )
    return subsequent_mask
