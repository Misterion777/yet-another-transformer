import torch
from torch.utils.data import Dataset

from src.constants import SEQ_LEN
from .dictionary import Dictionary

# Processed from: https://www.kaggle.com/datasets/blessondensil294/friends-tv-series-screenplay-script?resource=download
class FriendsDialog(Dataset):
    def __init__(self, csv_path: str, dictionary: Dictionary,seq_len=SEQ_LEN, build_dict=False):
        pass

    
    def __getitem__(self, index):        
        return 0,0
