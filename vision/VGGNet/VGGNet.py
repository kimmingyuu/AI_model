import torch
from torch import nn

cfgs = { "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
         "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
         "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
         "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"] }

class VGG(nn.Module):
    def __init__(self, cfg, num_class=1000, drop_p=0.5):
        super().__init__()
        # features
        self.conv3_fisrt = nn.Conv2d(3, 64, 3, padding=1)
        self.conv3_64 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_128_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_128 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_256_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_256 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_512_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv3_512 = nn.Conv2d(512, 512, 3, padding=1)
        
        # classifier
        self.fc_4096_first = nn.Linear(512 * 7 * 7, 4096)
        self.fc_4096 = nn.Linear(4096, 4096)
        self.fc_1000 = nn.Linear(4096, num_class)


        # etc        
        self.averagePool = nn.AdaptiveAvgPool2d((7,7)) # avg pooling that output is 7x7
        self.maxpool = nn.MaxPool2d(2)
        self.activation = nn.ReLU()
        self.drop_out = nn.Dropout(p=drop_p)


        self.features = nn.Sequential(self.conv3_fisrt,
                                      self.activation,
                                      self.conv3_64,
                                      self.activation,
                                      self.maxpool,
                                      self.conv3_128_1,
                                      self.activation,
                                      self.conv3_128,
                                      self.activation,
                                      self.maxpool,
                                      self.conv3_256_1,
                                      self.activation,
                                      self.conv3_256,
                                      self.activation,
                                      self.conv3_256,
                                      self.activation,
                                      self.maxpool,
                                      self.conv3_512_1,
                                      self.activation,
                                      self.conv3_512,
                                      self.activation,
                                      self.conv3_512,
                                      self.activation,
                                      self.maxpool,
                                      self.conv3_512,
                                      self.activation,
                                      self.conv3_512,
                                      self.activation,
                                      self.conv3_512,
                                      self.activation,
                                      self.maxpool)
        self.classifier = nn.Sequential(self.fc_4096_first,
                                        self.activation,
                                        self.drop_out,
                                        self.fc_4096,
                                        self.activation,
                                        self.drop_out,
                                        self.fc_1000)
    def forward(self, x):
        # features
        # x = self.features(x)
        x = self.conv3_fisrt(x)
        x = self.activation(x)
        x = self.conv3_64(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv3_128_1(x)
        x = self.activation(x)
        x = self.conv3_128(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv3_256_1(x)
        x = self.activation(x)
        x = self.conv3_256(x)
        x = self.activation(x)
        x = self.conv3_256(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.conv3_512_1(x)
        x = self.activation(x)
        x = self.conv3_512(x)
        x = self.activation(x)
        x = self.conv3_512(x)
        x = self.activation(x)
        x = self.activation(x)

        x = self.conv3_512(x)
        x = self.activation(x)
        x = self.conv3_512(x)
        x = self.activation(x)
        x = self.conv3_512(x)
        x = self.activation(x)
        x = self.activation(x)

        x = self.averagePool(x)

        # flatten
        x = torch.flatten(x, 1)

        # classifier
        # x = self.classifier(x)
        x = self.fc_4096_first(x)
        x = self.activation(x)
        x = self.drop_out(x)
        x = self.fc_4096(x)
        x = self.activation(x)
        x = self.drop_out(x)
        x = self.fc_1000(x)

        return x

model = VGG(cfgs["D"])

from torchinfo import summary
summary(model, input_size=(1, 3,224,224), device='cpu')

x = torch.rand(1, 3, 224, 224)

print(model(x).shape)