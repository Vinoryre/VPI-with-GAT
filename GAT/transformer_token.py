import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTransformer(nn.Module):
    """
    将 ROI 切分的 patch token 送入 Transformer 提取特征
    """
    def __init__(self,
                 input_dim=512,
                 embed_dim=512,
                 num_heads=8,
                 num_layers=6,
                 dropout=0.1,
                 use_pooling=False
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_pooling = use_pooling

        # 线性投影, patch的token维度降维
        self.patch_proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # 位置编码, 假设patch数量最多为64
        self.pos_embed = nn.Parameter(torch.randn(1, 32768, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*8,
            dropout = dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        :param self:
        :param x:
        :return:
        """
        B, N, E_in = x.shape
        assert N <= self.pos_embed.shape[1], f"patch数量 {N} 超过最大支持 {self.pos_embed.shape[1]}"

        # 线性降维 + LayerNorm
        x = self.patch_proj(x)
        x = self.norm(x)

        # 加可训练向量,用于模型学习和训练区分不同的patch
        x = x + self.pos_embed[:, :N, :]

        # Transformer Encoder
        x = self.transformer(x)

        if self.use_pooling:
            # N个patch token 平均池化
            x = x.mean(dim=1)

        return x


if __name__ == "__main__":
    B, N, E_in = 1, 4096, 512
    dummy_input = torch.randn(B, N, E_in)

    model = PatchTransformer(input_dim=E_in, embed_dim=512)
    out = model(dummy_input)
    print("output shape:", out.shape)

