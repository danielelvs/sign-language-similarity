from torch import nn
from torchvision.models import resnet18

from networks.base import BaseNetwork


class Resnet18Network(BaseNetwork):
    name = "resnet18"
    model = None
    transforms = None
    image_size = (224, 224)


    def __init__(self, num_classes: int):
        super().__init__(num_classes)

        self.model = resnet18(pretrained=True)

        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )


    def get_model(self):
        return self.model


    def get_fc_layer(self):
        return self.model.fc


    def get_transformers(self):
        return self.transforms
