import torch
from torch.utils.data import Dataset


class EvaluateDataset(Dataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform


    def __getitem__(self, idx):
        image = torch.tensor(self.X[idx:idx+1], dtype=torch.float32)
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return len(self.X)
