from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as scheduler
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import math
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  #root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from src.constants import BATCH_SIZE, EMB_DIM, HIDDEN_DIM

from src.dataset import Dictionary, WikiText
from src.transformer import GeneratorTransformer

def get_backend():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = torch.device(get_backend())
print(f"Using device: {DEVICE}")


class TransformerTrainer:
    def __init__(self, model):
        self.model = model        

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=5e-4
        )
        self.scheduler = scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.95
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        print("Trainer initialized, printing model architecture:")
        print(self.model)


    def train(self,train_loader:DataLoader,val_loader:DataLoader,num_epochs: int = 10,save_path="runs/"):
        save_path = Path(save_path) / f"{datetime.now()}"
        self.writer = SummaryWriter(save_path)

        self.model = self.model.to(DEVICE)

        best_ppl = 10e8 # just large number
        for epoch in range(1, num_epochs):
            train_loss, train_ppl = self._train_epoch(train_loader,epoch)
            val_loss, val_ppl = self.test(val_loader)            
            self._write_epoch_info(epoch,train_loss,train_ppl,val_loss,val_ppl)

            self.scheduler.step()
            self.writer.add_scalar(
                "train/learning rate",
                self.scheduler.get_last_lr()[-1],
                epoch,
            )

            print(f"Epoch: {epoch:03d} Loss: {train_loss:.4f}; Valid PPL: {val_ppl:.4f}")
            # the less perplexity the better
            if best_ppl > val_ppl:
                torch.save(
                    self.model.state_dict(),
                    save_path
                    / f"checkpoint_{epoch:03d}_{val_ppl:.4f}.pt",
                )
                best_ppl = val_ppl


    def _write_epoch_info(self,epoch,train_loss,train_ppl,val_loss,val_ppl):
        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("train/ppl", train_ppl, epoch)            
        self.writer.add_scalar("val/loss", val_loss, epoch)
        self.writer.add_scalar("val/ppl", val_ppl, epoch)            



    def _train_epoch(self, loader: DataLoader,epoch_num:int):
        self.model.train()

        pbar = tqdm(loader, total=len(loader))
        total_loss = 0
        for data,target in pbar:  # Iterate in batches over the training dataset.
            self.optimizer.zero_grad()  # Clear gradients.
            data = data.to(DEVICE)
            target = target.to(DEVICE) # (batch_size,seq_len)

            out = self.model(data,target) 
            out = out.transpose(1,2) # Should have shape (batch_size,dict_size,seq_len)

            loss = self.criterion(out,target)
            curr_loss = loss.item()

            total_loss += curr_loss
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.

            pbar.set_description(f"Epoch #{epoch_num}. Loss: {curr_loss:.4f}, PPL: {math.exp(curr_loss):.4f}")
        mean_loss = total_loss / len(loader.dataset)
        mean_ppl = math.exp(mean_loss)
        return mean_loss, mean_ppl        

    def test(self, loader: DataLoader):
        self.model.eval()

        pbar = tqdm(loader, total=len(loader))
        total_loss = 0
        for data,target in loader:        
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            out = self.model(data,target) 

            loss = self.criterion(out,target)
            curr_loss = loss.item()
            total_loss += curr_loss
            pbar.set_description(f"Test set. Loss: {curr_loss:.4f}, PPL: {math.exp(curr_loss):.4f}")
        mean_loss = total_loss / len(loader.dataset)
        mean_ppl = math.exp(mean_loss)
        return mean_loss, mean_ppl
        

if __name__ == "__main__":
    wiki_dict = Dictionary()
    train_ds = WikiText("data/wikitext-2/wiki.train.tokens",wiki_dict,build_dict=True)
    val_ds = WikiText("data/wikitext-2/wiki.valid.tokens", wiki_dict)

    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=False)
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)

    print(f"Dictionary size: {len(wiki_dict)}")
    model = GeneratorTransformer(EMB_DIM,HIDDEN_DIM,dict_size=len(wiki_dict))

    trainer = TransformerTrainer(model)

    trainer.train(train_loader,val_loader)