import os
from models import Star
from torch.utils.data import Dataset

class StarDataset(Dataset):
    def __init__(self, rootDir, transform=None):
        self.files = [os.path.join(rootDir, file) for file in os.listdir(rootDir)]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        star = Star.load(self.files[idx])
        if self.transform:
            star = self.transform(star)
        return star
