import torch


def get_backend():
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return "mps"
        else:
            print(
                "MPS device detected, but it's not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
