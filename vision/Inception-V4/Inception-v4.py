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

class Inception_Stem(nn.Module):
    # Figure 3.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride=2),
                                   BasicConv2d(32, 32, kernel_size=3),
                                   BasicConv2d(32, 64, kernel_size=3, padding=1))
        self.branch3x3_pool = nn.MaxPool2d(3, stride=2)
        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

        self.branch7x7a = nn.Sequential(BasicConv2d(160, 64, kernel_size=1),
                                        BasicConv2d(64, 96, kernel_size=3))
        self.branch7x7b = nn.Sequential(BasicConv2d(160, 64, kernel_size=1),
                                        BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
                                        BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
                                        BasicConv2d(64, 96, kernel_size=3))

        self.branchpoola = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.branchpoolb = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = [self.branch3x3_pool(x),
             self.branch3x3_conv(x)]
        x = torch.cat(x, 1)

        x = [self.branch7x7a(x),
             self.branch7x7b(x)]
        x = torch.cat(x, 1)

        x = [self.branchpoola(x),
             self.branchpoolb(x)]
        x = torch.cat(x, 1)
        return x

class InceptionA(nn.Module):
    # Figure 4.
    def __init__(self, input_channels):
        super().__init__()

        self.branchpool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                        BasicConv2d(input_channels, 96, kernel_size=1))

        self.branch1x1 = BasicConv2d(input_channels, 96, kernel_size=1)

        self.branch3x3 = nn.Sequential(BasicConv2d(input_channels, 64, kernel_size=1),
                                       BasicConv2d(64, 96, kernel_size=3, padding=1))

        self.branch3x3dbl = nn.Sequential(BasicConv2d(input_channels, 64, kernel_size=1),
                                          BasicConv2d(64, 96, kernel_size=3, padding=1),
                                          BasicConv2d(96, 96, kernel_size=3, padding=1))

    def forward(self, x):
        x = [self.branchpool(x), self.branch1x1(x), self.branch3x3(x), self.branch3x3dbl(x)]
        return torch.cat(x, 1)

class ReductionA(nn.Module):
    # Figure 7.
    # The k, l, m, n numbers represent filter bank sizes which can be looked up in Table 1.
    def __init__(self, input_channels, k, l, m, n):
        super().__init__()

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)
        self.branch3x3dbl = nn.Sequential(BasicConv2d(input_channels, k, kernel_size=1),
                                          BasicConv2d(k, l, kernel_size=3, padding=1),
                                          BasicConv2d(l, m, kernel_size=3, stride=2))

    def forward(self, x):
        x = [self.branchpool(x), self.branch3x3(x), self.branch3x3dbl(x)]
        return torch.cat(x, 1)

class InceptionB(nn.Module):

    # Figure 5.
    def __init__(self, input_channels):
        super().__init__()

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(input_channels, 128, kernel_size=1))

        self.branch1x1 = BasicConv2d(input_channels, 384, kernel_size=1)

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))) # 논문은 여기도 1x7로 되어있음. 하지만 오류일 것으로 추정

        self.branch7x7dbl = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)))

    def forward(self, x):
        x = [self.branchpool(x), self.branch1x1(x), self.branch7x7(x), self.branch7x7dbl(x)]
        return torch.cat(x, 1)

class ReductionB(nn.Module):
    # Figure 8.
    def __init__(self, input_channels):
        super().__init__()

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2))

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2))

    def forward(self, x):
        x = [self.branchpool(x), self.branch3x3(x), self.branch7x7(x)]
        return torch.cat(x, 1)

class InceptionC(nn.Module):
    # Figure 6.
    def __init__(self, input_channels):
        super().__init__()

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 256, kernel_size=1))

        self.branch1x1 = BasicConv2d(input_channels, 256, kernel_size=1)

        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3a = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3b = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl = nn.Sequential(
            BasicConv2d(input_channels, 384, kernel_size=1),
            BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)))
        self.branch3x3dbla = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dblb = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        branchpool = self.branchpool(x)

        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3(x)
        branch3x3 = [self.branch3x3a(branch3x3),
                     self.branch3x3b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl(x)
        branch3x3dbl = [self.branch3x3dbla(branch3x3dbl),
                        self.branch3x3dblb(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branchpool]

        return torch.cat(outputs, 1)

class InceptionV4(nn.Module):
    # Figure 9.
    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, class_nums=1000):
        super().__init__()
        self.stem = Inception_Stem()
        self.inception_a = nn.Sequential(*[InceptionA(384) for _ in range(A)])
        self.reduction_a = ReductionA(384, k, l, m, n)
        self.inception_b = nn.Sequential(*[InceptionB(1024) for _ in range(B)])
        self.reduction_b = ReductionB(1024)
        self.inception_c = nn.Sequential(*[InceptionC(1536) for _ in range(C)])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout2d(0.2)
        self.linear = nn.Linear(1536, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x,1)
        x = self.linear(x)
        return x
    

model = InceptionV4(4, 7, 3)
summary(model, input_size=(2,3,299,299), device='cpu')

x = model(torch.randn(2,3,299,299))
print(x.shape)