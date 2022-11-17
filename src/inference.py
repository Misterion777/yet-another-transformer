import argparse

import torch

from src.constants import EMB_DIM, HIDDEN_DIM
from src.datasets.dictionary import Dictionary
from src.datasets.friends import FriendsDialog
from src.datasets.wiki_text import WikiText
from src.model.transformer import GeneratorTransformer
from src.utils import get_backend

DEVICE = torch.device(get_backend())
print(f"Using device: {DEVICE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start transformer training on one of the datasets"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/checkpoint_001_1.8447.pt",
        type=str,
        help="path to trained checkpoint of the model",
    )
    parser.add_argument(
        "--dataset",
        default="wikitext",
        type=str,
        choices=["wikitext", "friends"],
        help="name of the dataset to train on",
    )
    parser.add_argument(
        "--prompt",
        default=" The most significant benefit of the use of the technique is",
        type=str,
        help="prompt to generate from",
    )
    args = parser.parse_args()

    # Build dictionary
    ds_dict = Dictionary()
    if args.dataset == "wikitext":
        train_ds = WikiText(
            "data/wikitext-2/wiki.train.tokens", ds_dict, build_dict=True
        )
    elif args.dataset == "friends":
        train_ds = FriendsDialog(
            "data/friends/train.txt", ds_dict, build_dict=True
        )
    else:
        raise ValueError("Unknown dataset!")

    model = GeneratorTransformer(EMB_DIM, HIDDEN_DIM, dict_size=len(ds_dict))
    model = model.to(DEVICE)

    model.load_checkpoint(args.checkpoint, DEVICE)

    tokens = ds_dict.tokenize(args.prompt)
    data = ds_dict.tokens2id(tokens)
    data = torch.tensor(data, device=DEVICE).unsqueeze(
        0
    )  # add batch dimension

    generated = model.generate(data, ds_dict.bos_id)
    data = data[:, 1:]  # Ignore BOS
    generated = generated.cpu().numpy()
    prompt = ds_dict.ids2tokens(data[-1])
    answer = ds_dict.ids2tokens(generated[-1])
    print(f"Prompt:{prompt}\nAnswer:{answer}\n")
