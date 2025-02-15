# modules of nn
# nn.Sequential()
# nn.Conv2d()
# nn.BatchNorm2d()
# nn.ReLU()
# nn.MaxPool2d
# nn.AvgPool2d
# nn.Linear
# nn.Dropout

import torch
from torch import nn

# Class : BasicConv2d(), Inception(), InceptionAux(), Inception_V1()
class BasicConv2d(nn.modules):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Inception(nn.modules):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class InceptionAux(nn.modules):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
class Inception_V1(nn.modules):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
# 
model = Inception_V1()
x = model(torch.rand(2, 3, 224, 224))