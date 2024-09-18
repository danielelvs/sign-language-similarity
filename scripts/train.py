import numpy as np
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
  """
  Data class for Siamese Network training.
  """

  def __init__(self, X, y, size=None, transform=None):
    """
    Initialize the dataset with the images and their labels.

    Args:
    - X: Images.
    - y: Labels.
    - transform: Transform to apply to the images.
    """

    self.X = X
    self.y = y
    self.transform = transform
    self.size = size if size is not None else X.shape[0]

  def __getitem__(self, idx):
    """
    Selects two random images and returns the data.

    Args:
    - idx: Dummy variable.
    Returns:
    - tuple: Two random images and their labels.
    """

    idx1 = np.random.randint(0, len(self.X))
    # idx1 = random.randint(0, len(self.X) - 1)
    # image1 = Image.fromarray(self.X[idx1], mode='L')
    image1 = torch.tensor(self.X[idx1:idx1+1], dtype=torch.float32)
    # image1 = self.X[idx1][np.newaxis]
    label1 = self.y[idx1]

    # Sample a positive or negative image
    should_match = np.random.choice([True, False])#random.randint(0, 1) # we need to approximately 50% of images to be in the same class
    mask_candidates = (self.y == label1) if should_match else (self.y != label1)
    mask_candidates[idx1] = False # Prevent sampling the same sample
    idx_candidates = np.where(mask_candidates)[0]
    idx2 = np.random.choice(idx_candidates)

    # if should_match:
    #     while True: # look until the same class image is found
    #         idx2 = random.randint(0, len(self.X) - 1)
    #         if label1 == self.y[idx2]:
    #             break
    # else:
    #     while True: # look until a different class image is found
    #         idx2 = random.randint(0, len(self.X) - 1)
    #         if label1 != self.y[idx2]:
    #             break

    # image2 = Image.fromarray(self.X[idx2], mode='L')
    image2 = torch.tensor(self.X[idx2:idx2+1], dtype=torch.float32)
    # image2 = self.X[idx2][np.newaxis]
    label2 = self.y[idx2]

    if self.transform is not None:
        image1 = self.transform(image1)
        image2 = self.transform(image2)

    return image1, image2, torch.tensor(1 - should_match, dtype=torch.float32)

  def __len__(self):
    """ Returns the size of the dataset."""

    return self.size
