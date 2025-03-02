import torch
from torch import nn
from torchinfo import summary

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
                                        nn.BatchNorm2d(out_channels, eps=0.001),
                                        nn.ReLU())
        
    def forward(self, x):
        x = self.conv_block(x)
        return x
    
class InceptionA(nn.Module): # Figure 5
    def __init__(self, in_channels, pool_features):
        super().__init__()

        self.branch3x3dbl = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1), # dbl = double
                                          BasicConv2d(64, 96, kernel_size=3, padding=1),
                                          BasicConv2d(96, 96, kernel_size=3, padding=1))
        
        self.branch3x3 = nn.Sequential(BasicConv2d(in_channels, 48, kernel_size=1),
                                       BasicConv2d(48, 64, kernel_size=3, padding=1))
        
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         BasicConv2d(in_channels, pool_features, kernel_size=1))
        
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        x = [self.branch3x3dbl(x), self.branch3x3(x), self.branch_pool(x), self.branch1x1(x)]
        return torch.cat(x, 1)
    
class InceptionB(nn.Module): # Figure 6
    def __init__(self, in_channels, channels_7x7):
        super().__init__()

        self.branch7x7dbl = nn.Sequential(BasicConv2d(in_channels, channels_7x7, kernel_size=1),
                                          BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1,7), padding=(0,3)),
                                          BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7,1), padding=(3,0)),
                                          BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1,7), padding=(0,3)),
                                          BasicConv2d(channels_7x7, 192, kernel_size=(7,1), padding=(3,0)))
        
        self.branch7x7 = nn.Sequential(BasicConv2d(in_channels, channels_7x7, kernel_size=1),
                                       BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1,7), padding=(0,3)),
                                       BasicConv2d(channels_7x7, 192, kernel_size=(7,1), padding=(3,0)))
        
        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         BasicConv2d(in_channels, 192, kernel_size=1))
        
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        x = [self.branch7x7dbl(x), self.branch7x7(x), self.branch_pool(x), self.branch1x1(x)]
        return torch.cat(x, 1)
    
class InceptionC(nn.Module): # Figure 7
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3dbl = nn.Sequential(BasicConv2d(in_channels, 448, kernel_size=1),
                                          BasicConv2d(448, 384, kernel_size=3, padding=1))
        self.branch3x3dbla = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dblb = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         BasicConv2d(in_channels, 192, kernel_size=1))

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

    def forward(self, x):
        branch3x3dbl = self.branch3x3dbl(x)
        branch3x3dbl = [self.branch3x3dbla(branch3x3dbl),
                        self.branch3x3dblb(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch3x3 = self.branch3x3(x)
        branch3x3 = [self.branch3x3a(branch3x3),
                     self.branch3x3b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        branch_pool = self.branch_pool(x)

        branch1x1 = self.branch1x1(x)

        outputs = [branch3x3dbl, branch3x3, branch_pool, branch1x1]
        return torch.cat(outputs,1)
    
class ReductionA(nn.Module): # Bottleneck 피하면서 grid-size 줄이기
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3dbl = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1),
                                          BasicConv2d(64, 96, kernel_size=3, padding=1),
                                          BasicConv2d(96, 96, kernel_size=3, stride=2))

        self.branch3x3 = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1),
                                       BasicConv2d(64, 384, kernel_size=3, stride=2))


        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = [self.branch3x3dbl(x), self.branch3x3(x), self.branch_pool(x)]
        return torch.cat(x,1)
    
class ReductionB(nn.Module): # Bottleneck 피하면서 grid-size 줄이기
    def __init__(self, in_channels):
        super().__init__()

        self.branch3x3dbl = nn.Sequential(BasicConv2d(in_channels, 192, kernel_size=1),
                                          BasicConv2d(192, 192, kernel_size=3, padding=1),
                                          BasicConv2d(192, 192, kernel_size=3, stride=2))

        self.branch3x3 = nn.Sequential(BasicConv2d(in_channels, 192, kernel_size=1),
                                       BasicConv2d(192, 320, kernel_size=3, stride=2))

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = [self.branch3x3dbl(x), self.branch3x3(x), self.branch_pool(x)]
        return torch.cat(x,1)
    
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.fc1.stddev = 0.001
        self.fc2 = nn.Linear(1024, num_classes)
        self.fc2.stddev = 0.001

    def forward(self, x):
        x = self.avgpool1(x)
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class Inception_V3(nn.Module):
    def __init__(self, num_classes = 1000, use_aux = True, init_weights = None, drop_p = 0.5):
        super().__init__()

        self.use_aux = use_aux

        self.conv1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv1b = BasicConv2d(32, 32, kernel_size=3)
        self.conv1c = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2a = BasicConv2d(64, 80, kernel_size=3)
        self.conv2b = BasicConv2d(80, 192, kernel_size=3, stride=2)
        self.conv2c = BasicConv2d(192, 288, kernel_size=3, padding=1)

        self.inception3a = InceptionA(288, pool_features=64)
        self.inception3b = InceptionA(288, pool_features=64)
        self.inception3c = ReductionA(288)

        self.inception4a = InceptionB(768, channels_7x7=128)
        self.inception4b = InceptionB(768, channels_7x7=160)
        self.inception4c = InceptionB(768, channels_7x7=160)
        self.inception4d = InceptionB(768, channels_7x7=192)
        if use_aux:
            self.aux = InceptionAux(768, num_classes)
        else:
            self.aux = None
        self.inception4e = ReductionB(768)

        self.inception5a = InceptionC(1280)
        self.inception5b = InceptionC(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.conv1c(x)
        x = self.maxpool1(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv2c(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux is not None and self.training:
            aux = self.aux(x)
        else:
            aux = None  

        x = self.inception4e(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, aux
    
model = Inception_V3()

summary(model, input_size=(2, 3, 299, 299), device='cpu')

x, x_aux = model(torch.rand(2, 3, 299, 299))
print(x.shape)
print(x_aux.shape)