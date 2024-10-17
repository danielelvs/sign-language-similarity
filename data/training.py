import numpy as np
import torch
from torch.utils.data import Dataset


class TrainingData(Dataset):

    def __init__(self, X, y, size=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.size = size if size is not None else X.shape[0]


    def __getitem__(self, idx):
        idx1 = np.random.randint(0, len(self.X))
        image1 = torch.tensor(self.X[idx1], dtype=torch.float32)
        label1 = self.y[idx1]

        should_match = np.random.choice([True, False])#random.randint(0, 1) # we need to approximately 50% of images to be in the same class
        mask_candidates = (self.y == label1) if should_match else (self.y != label1)
        mask_candidates[idx1] = False # Prevent sampling the same sample
        idx_candidates = np.where(mask_candidates)[0]

        idx2 = np.random.choice(idx_candidates)
        image2 = torch.tensor(self.X[idx2], dtype=torch.float32)
        label2 = self.y[idx2]

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(1 - should_match, dtype=torch.float32)


    def __len__(self):
        return self.size
