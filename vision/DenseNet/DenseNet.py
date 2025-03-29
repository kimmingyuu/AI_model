import torch
import torch.nn as nn
from torchinfo import summary

class Bottleneck(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()

        self.residual = nn.Sequential(nn.BatchNorm2d(in_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels, 4*k, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(4*k),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(4*k, k, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.residual(x)], 1)
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.transition = nn.Sequential(nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                        nn.AvgPool2d(2))

    def forward(self, x):
        return self.transition(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0< theta ≤1)
class DenseNet(nn.Module):
    def __init__(self, num_block_list, growth_rate, reduction=0.5, num_class=1000):
        super().__init__()
        self.k = growth_rate

        inner_channels = 2 * self.k

        self.conv1 = nn.Sequential(nn.Conv2d(3, inner_channels, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(inner_channels),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        layers = []
        for num_blocks in num_block_list[:-1]:
            layers += [self.make_dense_block(inner_channels, num_blocks)]
            inner_channels +=  num_blocks * self.k

            out_channels = int(reduction * inner_channels)
            layers += [Transition(inner_channels, out_channels)]
            inner_channels = out_channels
        
        layers += [self.make_dense_block(inner_channels, num_block_list[-1])]
        inner_channels += num_block_list[-1] * self.k

        layers += [nn.BatchNorm2d(inner_channels)]
        layers += [nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.linear(output)
        return output
    
    def make_dense_block(self, in_channels, nblocks):
        dense_block = []
        for _ in range(nblocks):
            dense_block += [ Bottleneck(in_channels, self.k) ]
            in_channels += self.k
        return nn.Sequential(*dense_block)
    
def densenet121(**kwargs):
    return DenseNet([6,12,24,16], growth_rate=32, **kwargs)

def densenet169(**kwargs):
    return DenseNet([6,12,32,32], growth_rate=32, **kwargs)

def densenet201(**kwargs):
    return DenseNet([6,12,48,32], growth_rate=32, **kwargs)

def densenet264(**kwargs):
    return DenseNet([6,12,64,48], growth_rate=32, **kwargs)\
    
model = densenet264()
summary(model, input_size=(2, 3, 224, 224), device='cpu')

x = torch.randn(2, 3, 224, 224)
print(model(x).shape)