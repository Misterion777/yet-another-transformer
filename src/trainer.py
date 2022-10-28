import torch
import torch.optim.lr_scheduler as scheduler

from src.dataset import Dictionary, WikiText

def get_backend():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():        
        return "mps"        
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = torch.device(get_backend())
print(f"Using device: {DEVICE}")

class TransformerTrainer:
    def __init__(self,model):
        self.model = model

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=5e-4
        )
        self.scheduler = scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.95
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        print(self.model)



    
    
