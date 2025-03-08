import torch
from torch import nn
from torchinfo import summary

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, inner_channerls, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, inner_channerls, 3, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channerls),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channerls, inner_channerls, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channerls))
        
        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None: # projection shortcut : 1x1 conv with stride 2
            shortcut = self.projection(x)
        else:
            shortcut = x # identity shortcut
        
        out = self.relu(residual + shortcut)
        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution
    # while original implementation places the stride at the first 1x1 convolution
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion = 4
    def __init__(self, in_channels, inner_channels, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, inner_channels, 1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias=False),
                                      nn.BatchNorm2d(inner_channels * self.expansion))
        
        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x
        
        out = self.relu(residual + shortcut)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block_list, num_class=1000, zero_init_residual = True):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self.make_stage(block, 64, num_block_list[0], stride=1)
        self.stage2 = self.make_stage(block, 128, num_block_list[1], stride=2)
        self.stage3 = self.make_stage(block, 256, num_block_list[2], stride=2)
        self.stage4 = self.make_stage(block, 512, num_block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_class)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%p according to https://arxiv.org/abs/1706.02677
        # 아무것도 보태지 않은 상태로 학습을 시작해서 뭘 보태면 좋을지를 알아내라!
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.residual[-1].weight, 0)
    
    def make_stage(self, block, inner_channels, num_blocks, stride = 1):
        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            # stride = 2 면 여기로 무조건 들어옴 즉, stage 2,3,4 는 무조건 여기로 들어옴. (Basic block, BottleNeck 상관없이)
            # stride = 1 이여도 채널 수가 다르면 여기로 들어옴 (resoltion은 그대로, 채널 수만 늘어나는 때)
            # 즉, Basic block 쓰는 18, 34-layer의 stage 1에서만 else로 가고 BottleNeck 쓰는 50, 101, 152-layer는 모든 stage에서 항상 여기로 들어옴
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(inner_channels * block.expansion)) # projection shortcut
        else:
            projection = None # identity shortcut
        
        layers = []
        layers += [block(self.in_channels, inner_channels, stride, projection)]
        self.in_channels = inner_channels * block.expansion
        for _ in range(1, num_blocks): # 처음 block만 projection shortcut 나머진 identity shortcut
            layers += [block(self.in_channels, inner_channels)]
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

model = resnet18()
summary(model, input_size=(2, 3, 224, 224), device='cpu')

x = torch.rand(2, 3, 224, 224)
print(model(x).shape)