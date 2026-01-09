import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewFeature(nn.Module):
    """
    多视角 patch 特征提取器
    """
    def __init__(self, input_dim=512, embed_dim=256, num_views=3, dropout=0.1):
        super(MultiViewFeature, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_views = num_views

        # 定义 num_views 个独立的 MLP, 每个 MLP 当作是一个视角
        self.view_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_views)
        ])

    def forward(self, x):
        """

        :param x: [B, N, C]
        :return: [B, N, embed_dim * num_views]
        """
        view_outputs = []

        for i, mlp in enumerate(self.view_mlps):
            # 每个视角独立映射
            view_feat = mlp(x)
            view_outputs.append(view_feat)

        # 最后一个维度拼接不同视角
        out = torch.cat(view_outputs, dim=-1)
        return out


if __name__ == '__main__':
    B, N, C = 2, 4096, 512
    dummy_input = torch.randn(B, N, C)

    model = MultiViewFeature(input_dim=C, embed_dim=256, num_views=3)
    output = model(dummy_input)
    print("Input shape: ", dummy_input.shape)
    print("Output shape: ", output.shape)
