import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """
    Patch 级别特征融合模块
    输入为 :
        x_trans: [B, N, D] Transformer token
        x_view: [B, N, D] Multi-view token

    输出为:
        x_fused: [B, N, D]
    """
    def __init__(self,
                 trans_dim=512,
                 view_dim=768,
                 embed_dim=512):
        super().__init__()

        self.embed_dim = embed_dim

        # 可训练的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # 维度对齐
        self.view_proj = nn.Linear(view_dim, embed_dim)

        # fusion后线性投影
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_trans, x_view):
        # 维度对齐
        x_view = self.view_proj(x_view)

        assert x_trans.shape == x_view.shape, "Fusion inputs must have same shape"

        alpha = torch.sigmoid(self.alpha)

        x_fused = alpha * x_trans + (1 - alpha) * x_view
        # x_fused = alpha * x_trans

        x = self.proj(x_fused)

        return x_fused


if __name__ == "__main__":
    B, N, D = 2, 4096, 512
    x_trans = torch.randn(B, N, D)
    x_view = torch.randn(B, N, 768)

    fusion = FeatureFusion(embed_dim=D)
    out = fusion(x_trans, x_view)

    print("Output shape: ", out.shape)
