import torch
from torch.utils.data import Dataset

class TSPytorchDataset(Dataset):
    def __init__(self, sequence, labels):
        self.sequence = torch.tensor(sequence).float()
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx], self.labels[idx]