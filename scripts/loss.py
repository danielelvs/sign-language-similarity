
import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
  """ Contrastive loss function. """

  def __init__(self, margin=2.0):
    """
    Initialize the loss function.

    Args:
    - margin: Margin.
    """
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, output1, output2, label):
    """
    Forward pass.

    Args:
    - output1: First network output.
    - output2: Second network output.
    - label: Label (binary).

    Returns:
    - Loss.
    """
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean(
      (1 - label) * torch.pow(euclidean_distance, 2) +
      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
    )

    return loss_contrastive
