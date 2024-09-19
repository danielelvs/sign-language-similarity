
import torch.nn as nn

class Siamese(nn.Module):
  """
  Siamese Network.
  """

  def __init__(self, feat_dim=512):
    """
    Initialize the Siamese Network.
    """

    super(Siamese, self).__init__()

    self.cnn1 = nn.Sequential(
      nn.Conv2d(1, 96, kernel_size=11, stride=4),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(3, stride=2),

      nn.Conv2d(96, 256, kernel_size=5, stride=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2),

      nn.Conv2d(256, 384, kernel_size=3, stride=1),
      nn.ReLU(inplace=True)
    ) # 1x1x384 for 100x100 input

    self.fc1 = nn.Sequential(
      nn.Linear(384, feat_dim),
    )

  def forward_once(self, x):
    """
    Forward pass once.

    Args:
    - x: Input.

    Returns:
    - Output.
    """
    output = self.cnn1(x)
    output = output.view(output.size()[0], -1) # batches x feat_dim
    output = self.fc1(output)
    return output

  def forward(self, input1, input2):
    """
    Forward pass.

    Args:
    - input1: First input.
    - input2: Second input

    Returns:
    - Output1.
    - Output2.
    """
    output1 = self.forward_once(input1)
    output2 = self.forward_once(input2)
    return output1, output2
