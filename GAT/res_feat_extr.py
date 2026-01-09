import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from VPI_Resnet_18.Resnet_18.model import Stem, Stage


class ResNet18_FeatureExtractor(nn.Module):
    """
    3D ResNet-18 特征提取器
    输出为 4096 维度特征
    """

    def __init__(self, weight_path=None, freeze=True):
        super().__init__()

        # Backbone
        self.stem = Stem()
        self.stage1 = Stage(64, 64, num_blocks=2, stride=1)
        self.stage2 = Stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = Stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = Stage(256, 512, num_blocks=2, stride=2)

        # 加载权重
        if weight_path is not None:
            self._load_backbone_weights(weight_path)

        # 是否冻结
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _load_backbone_weights(self, weight_path):
        """
        加载 backbone 权重, 忽略 fc
        :param self:
        :param weight_path:
        :return:
        """

        state_dict = torch.load(weight_path, map_location="cpu")

        # 删除 fc 层权重
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("fc.")
        }

        self.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded ResNet backbone weights from {weight_path}")

    def forward(self, x):
        """
        :param self:
        :param x: [B, 1, D, H, W]
        :return: [B, 4096]
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.avg_pool3d(x, 4)
        feat = torch.flatten(x, 1)

        return feat


if __name__ == "__main__":
    input = torch.randn(4, 1, 128, 128, 128)
    model = ResNet18_FeatureExtractor()
    output = model(input)
    print(output.shape)
