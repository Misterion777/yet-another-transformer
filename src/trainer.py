import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim.lr_scheduler as scheduler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from src.constants import BATCH_SIZE, EMB_DIM, HIDDEN_DIM
from src.datasets.dictionary import Dictionary
from src.datasets.friends import FriendsDialog
from src.datasets.wiki_text import WikiText
from src.model.transformer import GeneratorTransformer
from src.utils import get_backend

DEVICE = torch.device(get_backend())
print(f"Using device: {DEVICE}")


class TransformerTrainer:
    def __init__(self, model: nn.Module, dictionary: Dictionary):
        self.model = model
        self.dictionary = dictionary

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=5e-4
        )
        self.scheduler = scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.95
        )
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.dictionary.pad_id
        ).to(DEVICE)
        print("Trainer initialized, printing model architecture:")
        print(self.model)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        save_path="runs/",
    ):
        save_path = Path(save_path) / f"{datetime.now()}"
        self.writer = SummaryWriter(save_path)

        self.model = self.model.to(DEVICE)

        best_ppl = 10e8  # just large number
        for epoch in range(1, num_epochs):
            train_loss, train_ppl, train_acc = self._train_epoch(
                train_loader, epoch
            )
            val_loss, val_ppl, val_acc = self.test(val_loader)
            self._write_epoch_info(
                epoch, "train", loss=train_loss, ppl=train_ppl, acc=train_acc
            )
            self._write_epoch_info(
                epoch, "val", loss=val_loss, ppl=val_ppl, acc=val_acc
            )

            self.scheduler.step()
            self.writer.add_scalar(
                "train/learning rate",
                self.scheduler.get_last_lr()[-1],
                epoch,
            )

            print(
                f"Epoch: {epoch:03d} Loss: {train_loss:.4f}; Valid PPL: {val_ppl:.4f}"
            )
            # the less perplexity the better
            if best_ppl > val_ppl:
                torch.save(
                    self.model.state_dict(),
                    save_path / f"checkpoint_{epoch:03d}_{val_ppl:.4f}.pt",
                )
                best_ppl = val_ppl

    def _write_epoch_info(self, epoch: int, stage: str, **kwargs):
        for key, item in kwargs.items():
            self.writer.add_scalar(f"{stage}/{key}", item, epoch)

    def _train_epoch(self, loader: DataLoader, epoch_num: int):
        self.model.train()

        pbar = tqdm(loader, total=len(loader))
        total_loss = 0
        accuracies = []
        for (
            data,
            target,
        ) in pbar:  # Iterate in batches over the training dataset.
            self.optimizer.zero_grad()  # Clear gradients.
            data = data.to(DEVICE)
            target = target.to(DEVICE)  # (batch_size,seq_len)

            input_mask = data != self.dictionary.pad_id
            out = self.model(data, target, input_mask)

            # out = out[:, 1:-1, :] # Ignore BOS and EOS when calculating loss
            # target = target[:, 1:-1] # Ignore BOS and EOS when calculating loss

            out = out.transpose(
                1, 2
            )  # Should have shape (batch_size,dict_size,seq_len)

            loss = self.criterion(out, target)
            curr_loss = loss.item() / len(data)

            total_loss += loss.item()
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.

            # Per-token accuracy in the current training batch
            accuracy = (
                torch.sum(out.argmax(dim=1) == target) / torch.numel(target)
            ).item()

            accuracies.append(accuracy)

            pbar.set_description(
                f"Epoch #{epoch_num}. Loss: {curr_loss:.4f}, PPL: {np.exp(curr_loss):.4f} ACC: {accuracy:.4f}"
            )
        mean_loss = total_loss / len(loader.dataset)
        mean_ppl = np.exp(mean_loss)
        return mean_loss, mean_ppl, np.mean(accuracies)

    def test(self, loader: DataLoader):
        self.model.eval()

        pbar = tqdm(loader, total=len(loader))
        total_loss = 0
        accuracies = []
        for data, target in pbar:
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            input_mask = data != self.dictionary.pad_id

            out = self.model(data, target, input_mask)
            out = out.transpose(1, 2)

            loss = self.criterion(out, target)
            curr_loss = loss.item() / len(data)
            total_loss += loss.item()

            # Rough estimate of per-token accuracy in the current training batch
            accuracy = (
                torch.sum(out.argmax(dim=1) == target) / torch.numel(target)
            ).item()
            accuracies.append(accuracy)

            pbar.set_description(
                f"Test set. Loss: {curr_loss:.4f}, PPL: {np.exp(curr_loss):.4f} ACC: {accuracy:.4f}"
            )
        mean_loss = total_loss / len(loader.dataset)
        mean_ppl = np.exp(mean_loss)
        return mean_loss, mean_ppl, np.mean(accuracies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start transformer training on one of the datasets"
    )
    parser.add_argument(
        "--dataset",
        default="friends",
        type=str,
        choices=["wikitext", "friends"],
        help="name of the dataset to train on",
    )
    args = parser.parse_args()

    ds_dict = Dictionary()
    if args.dataset == "wikitext":
        train_ds = WikiText(
            "data/wikitext-2/wiki.train.tokens", ds_dict, build_dict=True
        )
        val_ds = WikiText("data/wikitext-2/wiki.valid.tokens", ds_dict)
        test_ds = WikiText("data/wikitext-2/wiki.test.tokens", ds_dict)
    elif args.dataset == "friends":
        train_ds = FriendsDialog(
            "data/friends/train.txt", ds_dict, build_dict=True
        )
        val_ds = FriendsDialog("data/friends/val.txt", ds_dict)
        test_ds = FriendsDialog("data/friends/test.txt", ds_dict)
    else:
        raise ValueError("Unknown dataset!")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dictionary size: {len(ds_dict)}")
    model = GeneratorTransformer(EMB_DIM, HIDDEN_DIM, dict_size=len(ds_dict))

    trainer = TransformerTrainer(model, ds_dict)

    trainer.train(train_loader, val_loader)
    print("Training complete, running evaluation on test set.")
    mean_loss, mean_ppl, mean_acc = trainer.test(test_loader)
    print(f"Test set PPL: {mean_ppl}")
