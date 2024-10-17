from torch import nn
import torchvision.models as models
from model.base import BaseModel


class Resnet50Model(BaseModel):
    name = "Resnet50"
    model = None
    transforms = None
    image_size = (224, 224)


    def __init__(self, num_classes: int):
        super(Resnet50Model, self).__init__()

        self.model = models.resnet50(pretrained=True)

        # self.model.flatten = nn.Flatten()

        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )


    def forward_once(self, x):
        output = self.model(x)
        output = output.view(output.size()[0], -1) # batches x feat_dim
        output = self.model.fc(output)
        return output


    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


    def get_model(self):
        return self.model


    # def get_model_flatten(self):
    #     return self.model.flatten


    def get_fc_layer(self):
        return self.model.fc


    def get_transformers(self):
        return self.transforms
