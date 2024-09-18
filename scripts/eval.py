import torch
from torch.utils.data import Dataset

class EvalDataset(Dataset):
  """
  Data class for Siamese Network evaluation/validation.
  """

  def __init__(self, X, y, transform=None):
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

  def __getitem__(self, idx):
    """
    Selects a single image and returns the data.

    Args:
    - idx: Index of the image.
    Returns:
    - tuple: A single image and its label.
    """

    # image = Image.fromarray(self.X[index], mode='L')
    image = torch.tensor(self.X[idx:idx+1], dtype=torch.float32)
    label = self.y[idx]

    if self.transform is not None:
      image = self.transform(image)

    return image, label

  def __len__(self):
    """ Returns the size of the dataset."""

    return len(self.X)
