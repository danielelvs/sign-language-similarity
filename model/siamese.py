import torch.nn as nn

from model.base import BaseModel


class SiameseModel(nn.Module, BaseModel):
    name = "Siamese"
    model = None
    transforms = None
    image_size = (100, 100)


    def __init__(self, feat_dim=128):
        super(SiameseModel, self).__init__()

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
            nn.Flatten(),
            nn.Linear(384, feat_dim)
        )


    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1) # batches x feat_dim
        output = self.fc1(output)
        return output


    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


    def get_model(self):
        return self.model


    def get_fc_layer(self):
        return self.model.fc


    def get_transformers(self):
        return self.transforms


    def get_image_size(self):
        return self.image_size
