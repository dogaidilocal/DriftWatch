"""CIFAR ResNet-20 architecture.

Written to match the state_dict of the chenyaofo/pytorch-cifar-models
`cifar10_resnet20` checkpoint, so the published weights load with strict=True.
Defined locally rather than fetched through torch.hub, because torch.hub's
repository download is not reachable from this environment.

Architecture: conv1(3->16) -> 3 stages of 3 BasicBlocks at 16/32/64 channels
-> global average pool -> fc(64, 10).
"""
import torch.nn as nn


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CifarResNet(nn.Module):
    def __init__(self, blocks_per_stage=(3, 3, 3), num_classes=10):
        super().__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._stage(16, blocks_per_stage[0], 1)
        self.layer2 = self._stage(32, blocks_per_stage[1], 2)
        self.layer3 = self._stage(64, blocks_per_stage[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _stage(self, planes, n_blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, n_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(self.avgpool(x).flatten(1))


def resnet20(num_classes=10):
    return CifarResNet((3, 3, 3), num_classes)
