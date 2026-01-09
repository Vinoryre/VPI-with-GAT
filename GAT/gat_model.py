import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data


def build_spherical_graph_coords(patch_features, cube_size=16):
    """
    使用空间坐标构建球状GAT图
    :param patch_features: [64, F], 每个patch的特征向量
    :param cube_size: 立方体尺寸
    :return: PyG Data对象
    """
    F = patch_features.shape[1]
    device = patch_features.device  # tensor全部在同一个device

    # 生成每个patch的3D坐标 (z, y, x)
    coords = torch.stack(torch.meshgrid(
        torch.arange(cube_size),
        torch.arange(cube_size),
        torch.arange(cube_size),
        indexing='ij'
    ), dim=-1).reshape(-1, 3).float().to(device)

    # 虚拟中心
    center = torch.tensor([[cube_size / 2 - 0.5] * 3], device=device)

    # 计算每个点到中心的距离 立方体壳, L∞范数
    dists = torch.max(torch.abs(coords - center), dim=1).values

    # 按距离分层
    max_dist = int(dists.max().item()) + 1
    layer_distances = [0.5 + i for i in range(max_dist)]
    layer_idx_list = [torch.nonzero(dists==d).flatten() for d in layer_distances]

    # DEBUG
    # for i, idx in enumerate(layer_idx_list):
    #     print(f"Layer {i} :")
    #     print(idx.tolist())
    #     print("num nodes:", len(idx))
    #     print("-" * 40)

    # 构建节点特征
    x_center = torch.zeros(1, F, device=device)
    x = torch.cat([x_center, patch_features], dim=0)

    # 构建边
    edge_index = []

    # 同层最近邻数量
    num__nn_same_layer = 4

    # 邻层最近邻数量列表
    # 中心->L1, L1->L2, L2->L3, L3 及之后都是2
    num_nn_next_layer_list = [8, 7, 3]

    # 中心->第一层
    if len(layer_idx_list) > 0:
        layer0 = layer_idx_list[0]
        coords0 = coords[layer0]
        coords_center = center
        dist_center = torch.cdist(coords_center, coords0)
        for i, p in enumerate(layer0):
            src = 0
            nn = torch.topk(dist_center[0], k=num_nn_next_layer_list[0], largest=False).indices
            for nj in nn:
                dst = layer0[nj].item() + 1
                edge_index += [[src, dst], [dst, src]]

    # 每层处理加 邻层
    for i, layer in enumerate(layer_idx_list):
        idx = layer
        if len(idx) == 0:
            continue
        coords_l = coords[idx]

        # 同层最近邻
        dist_same = torch.cdist(coords_l, coords_l)
        for j, p in enumerate(idx):
            src = p.item() + 1
            nn = torch.topk(dist_same[j], k=num__nn_same_layer+1, largest=False).indices
            for nj in nn:
                if nj.item() != j:
                    dst = idx[nj].item() + 1
                    edge_index += [[src, dst]]

        # 邻层最近邻
        if i+1 < len(layer_idx_list):
            next_idx = layer_idx_list[i+1]
            if len(next_idx) > 0:
                coords_next = coords[next_idx]
                dist_next = torch.cdist(coords_l, coords_next)

                # 每层的邻层决定数量
                if i+1 <= len(num_nn_next_layer_list):
                    k = num_nn_next_layer_list[i+1-1]
                else:
                    k = 2

                for j, p in enumerate(idx):
                    src = p.item() + 1
                    nn = torch.topk(dist_next[j], k=min(k, len(next_idx)), largest=False).indices
                    for nj in nn:
                        dst = next_idx[nj].item() + 1
                        edge_index += [[src, dst], [dst, src]]

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()

    # 最外层节点 mask
    layer_mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
    if len(layer_idx_list) > 1:
        layer_mask[layer_idx_list[-1]+1] = True

    data = Data(x=x, edge_index=edge_index)
    data.layer_mask = layer_mask
    data.layer_idx_list = layer_idx_list

    return data


class GATModel(nn.Module):
    """
    使用 PyG 的 GAT模型, 用于处理融合后的patch特征图
    其中输入: Data(x=[N, D], edge_index=[2, E])
    输出: 图分类 [B, num_classes]
    """
    def __init__(self, in_channels=512, hidden_channels=512, out_channels=2,
                 num_layers=3, heads=8, dropout=0.1):
        super(GATModel, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.dropout = dropout

        # 第一层 GAT
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))

        # 中间层 GAT
        for _ in range(num_layers - 2):
            self.convs.append((GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout)))

        # 输出层
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout))

        # 图级分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels//2, out_channels)
        )

    def forward(self, x, edge_index, batch=None, layer_mask=None):
        """

        :param x: 节点特征 [N_total, D]
        :param edge_index: 图边索引 [2, E_total]
        :param batch: 节点到图的映射 [N_total]
        :return: [B, num_classes]
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级聚合
        if layer_mask is not None:
            if batch is not None:
                x_layer = x[layer_mask]
                batch_layer = batch[layer_mask]
                x = global_mean_pool(x_layer, batch_layer)
            else:
                x = x[layer_mask].mean(dim=0, keepdim=True)
        else:
            if batch is not None:
                x = global_mean_pool(x, batch)

        out = self.classifier(x)
        return out


if __name__ == '__main__':
    N = 4096
    patch_feats = torch.randn(N, 512)
    data = build_spherical_graph_coords(patch_feats)

    gat = GATModel()
    batch = torch.zeros(data.x.shape[0], dtype=torch.long) # TODO: 这里如果我要并行多张图, 应该需要考虑batch的设置
    classify = gat(data.x, data.edge_index, batch=batch, layer_mask=data.layer_mask)
    print(data)
    print("edge_index shape: ", data.edge_index.shape)
    print("classify shape: ", classify.shape)

