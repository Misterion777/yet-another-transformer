from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as scheduler
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  #root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from src.constants import BATCH_SIZE, EMB_DIM, HIDDEN_DIM

from src.datasets.dictionary import Dictionary
from src.datasets.wiki_text import WikiText
from src.transformer import GeneratorTransformer

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

    def load_checkpoint(self,load_path: str):        
        loaded = torch.load(load_path,map_location=DEVICE)
        self.model.load_state_dict(loaded)            

    def train(self,train_loader:DataLoader,val_loader:DataLoader,num_epochs: int = 10,save_path="runs/"):
        save_path = Path(save_path) / f"{datetime.now()}"
        self.writer = SummaryWriter(save_path)

        self.model = self.model.to(DEVICE)

        best_ppl = 10e8 # just large number
        for epoch in range(1, num_epochs):
            train_loss, train_ppl, train_acc = self._train_epoch(train_loader,epoch)
            val_loss, val_ppl, val_acc = self.test(val_loader)            
            self._write_epoch_info(epoch,"train",loss=train_loss,ppl=train_ppl,acc=train_acc)
            self._write_epoch_info(epoch,"val",loss=val_loss,ppl=val_ppl,acc=val_acc)

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


    def _write_epoch_info(self,epoch:int,stage:str,**kwargs):
        for key,item in kwargs.items():
            self.writer.add_scalar(f"{stage}/{key}", item, epoch)

    def _train_epoch(self, loader: DataLoader,epoch_num:int):
        self.model.train()

        pbar = tqdm(loader, total=len(loader))
        total_loss = 0
        accuracies = []
        for data,target in pbar:  # Iterate in batches over the training dataset.
            self.optimizer.zero_grad()  # Clear gradients.
            data = data.to(DEVICE)
            target = target.to(DEVICE) # (batch_size,seq_len)

            out = self.model(data,target) 
            
            out = out[:, 1:-1, :] # Ignore BOS and EOS when calculating loss
            target = target[:, 1:-1] # Ignore BOS and EOS when calculating loss
            
            out = out.transpose(1,2) # Should have shape (batch_size,dict_size,seq_len)

            loss = self.criterion(out,target)            
            curr_loss = loss.item() / len(data)

            total_loss += loss.item()
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.

            # Rough estimate of per-token accuracy in the current training batch
            accuracy = (torch.sum(out.argmax(dim=1) == target) / torch.numel(target)).item()

            accuracies.append(accuracy)

            pbar.set_description(f"Epoch #{epoch_num}. Loss: {curr_loss:.4f}, PPL: {np.exp(curr_loss):.4f} ACC: {accuracy:.4f}")
        mean_loss = total_loss / len(loader.dataset)
        mean_ppl = np.exp(mean_loss)
        return mean_loss, mean_ppl, np.mean(accuracies)

    def test(self, loader: DataLoader):
        self.model.eval()

        pbar = tqdm(loader, total=len(loader))
        total_loss = 0
        accuracies = []
        for data,target in pbar:        
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            out = self.model(data,target) 
            out = out.transpose(1,2)

            loss = self.criterion(out,target)
            curr_loss = loss.item() / len(data)
            total_loss += loss.item()

            # Rough estimate of per-token accuracy in the current training batch
            accuracy = (torch.sum(out.argmax(dim=1) == target) / torch.numel(target)).item()
            accuracies.append(accuracy)

            pbar.set_description(f"Test set. Loss: {curr_loss:.4f}, PPL: {np.exp(curr_loss):.4f} ACC: {accuracy:.4f}")
        mean_loss = total_loss / len(loader.dataset)
        mean_ppl = np.exp(mean_loss)
        return mean_loss, mean_ppl, np.mean(accuracies)
        

if __name__ == "__main__":
    wiki_dict = Dictionary()
    train_ds = WikiText("data/wikitext-2/wiki.train.tokens",wiki_dict,build_dict=True)
    val_ds = WikiText("data/wikitext-2/wiki.valid.tokens", wiki_dict)
    test_ds = WikiText("data/wikitext-2/wiki.test.tokens", wiki_dict)

    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=False)
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)
    test_loader = DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False)

    print(f"Dictionary size: {len(wiki_dict)}")
    model = GeneratorTransformer(EMB_DIM,HIDDEN_DIM,dict_size=len(wiki_dict))

    trainer = TransformerTrainer(model)

    mode = "inference"
    if mode == "train":
        trainer.train(train_loader,val_loader)
        print("Training complete, running evaluation on test set.")
        mean_loss, mean_ppl,mean_acc = trainer.test(test_loader)
        print(f"Test set PPL: {mean_ppl}")
    elif mode == "inference":
        trainer.load_checkpoint("checkpoints/checkpoint_001_1.8447.pt")
        for data,target in train_loader:
            generated = model.generate(data,wiki_dict.bos_id)
            data = data[:, 1:-1] # Ignore BOS and EOS
            generated = generated.cpu().numpy()
            prompt = wiki_dict.ids2tokens(data[-1])
            answer = wiki_dict.ids2tokens(generated[-1])
            print(f"Prompt:{prompt}\nAnswer:{answer}\n")
            break

