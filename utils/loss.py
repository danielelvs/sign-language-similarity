
import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
  """
  ContrastiveLoss is a custom loss function used for training models on tasks that involve learning similarity metrics, such as in contrastive learning.
  """

  def __init__(self, margin=2.0):
    """
    Initializes the ContrastiveLoss class with a specified margin.
    Args:
      margin (float, optional): The margin value for the contrastive loss. Defaults to 2.0.
    """

    super(ContrastiveLoss, self).__init__()
    self.margin = margin


  def forward(self, output1, output2, label):
    """
    Computes the contrastive loss between two outputs.
    Args:
      output1 (torch.Tensor): The first output tensor.
      output2 (torch.Tensor): The second output tensor.
      label (torch.Tensor): The label tensor indicating whether the outputs are similar (1) or dissimilar (0).
    Returns:
      torch.Tensor: The computed contrastive loss.
    """

    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean(
      (1 - label) * torch.pow(euclidean_distance, 2) +
      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
    )

    return loss_contrastive
