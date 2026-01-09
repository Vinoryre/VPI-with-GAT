import torch
import torch.nn as nn
import torch.nn.functional as F


# Stem块
class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()

        # 大卷积核
        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )

        # TODO 可选 BN/NOT 的对比如何
        self.bn = nn.BatchNorm3d(64)

        # Relu
        self.relu = nn.ReLU(inplace=True)

        # 池化
        self.pool = nn.MaxPool3d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# 残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 主分支
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # 残差跳跃
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 残差对叠层
class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride):
        super(Stage, self).__init__()
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1
            ))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class ThreeD_ResNet_18(nn.Module):
    def __init__(self, num_classes=2):
        super(ThreeD_ResNet_18, self).__init__()
        self.stem = Stem()
        self.stage1 = Stage(64, 64, num_blocks=2, stride=1)
        self.stage2 = Stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = Stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = Stage(256, 512, num_blocks=2, stride=2)
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool3d(x, 4)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out


if __name__ == "__main__":
    input = torch.randn(4, 1, 128, 128, 128)
    model = ThreeD_ResNet_18()
    x = model.stem(input)
    print("after stem:", x.shape)
    x = model.stage1(x)
    print("after stage1:", x.shape)
    x = model.stage2(x)
    print("after stage2:", x.shape)
    x = model.stage3(x)
    print("after stage3:", x.shape)
    x = model.stage4(x)
    print("after stage4:", x.shape)
    x = F.avg_pool3d(x, 4)
    print("after avg_pool3d:", x.shape)
    x = torch.flatten(x, 1)
    print("after flatten:", x.shape)
    out = model.fc(x)
