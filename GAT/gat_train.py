import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn .model_selection import KFold
from dataloader import LungNoduleDataset
from transformer_token import PatchTransformer
from multi_view_feature import MultiViewFeature
from fusion import FeatureFusion
from gat_model import build_spherical_graph_coords, GATModel
from torch_geometric.data import Batch


def count_labels(dataset, indices):
    """
    给定subset 的indices, 统计 pos/neg 数量
    :param dataset:
    :param indices:
    :return:
    """
    pos = 0
    neg = 0
    for idx in indices:
        _, label = dataset.samples[idx]
        if label == 1:
            pos += 1
        else:
            neg += 1
    return neg, pos


def train_one_epoch(
        patch_transformer,
        multi_view,
        fusion,
        gat,
        loader,
        criterion,
        optimizer,
        device
):
    """

    :param patch_transformer: patch 级别的 transform模型
    :param multi_view: patch级别多视角模型
    :param fusion: trans 和 multi_view 融合模型
    :param gat: patch当作节点的图注意力模型
    :param loader: dataloader
    :param criterion: 计算损失
    :param optimizer: 参数更新
    :param device:
    :return:
    """
    patch_transformer.train()
    multi_view.train()
    fusion.train()
    gat.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for i, (x, y) in enumerate(loader):
        x = x.to(device) # [B, 64, 32768]
        y = y.to(device)

        optimizer.zero_grad()

        # Patch-level 特征
        x_trans = patch_transformer(x)     # [B,4096,512]
        x_view = multi_view(x)             # [B,4096,768]
        x_fused = fusion(x_trans, x_view)  # [B,4096,512]

        # 构图
        data_list = []
        for i in range(x_fused.size(0)):
            data = build_spherical_graph_coords(x_fused[i])
            data.y = y[i].unsqueeze(0)
            data_list.append(data)

        batch_graph = Batch.from_data_list(data_list).to(device)

        # GAT
        logits = gat(
            batch_graph.x,
            batch_graph.edge_index,
            batch_graph.batch,
        )  # [B, 2]

        loss = criterion(logits, batch_graph.y)

        loss.backward()

        # DEBUG,梯度回传
        switch = False
        if switch:
            print("=== Grad Norms ===")
            if i >= 8 & i <= 10:
                for name, param in patch_transformer.named_parameters():
                    if param.grad is not None:
                        print("PatchTransformer", name, param.grad.norm().item())
                for name, param in multi_view.named_parameters():
                    if param.grad is not None:
                        print("MultiViewFeature", name, param.grad.norm().item())
                for name, param in fusion.named_parameters():
                    if param.grad is not None:
                        print("FusionModel", name, param.grad.norm().item())
                for name, param in gat.named_parameters():
                    if param.grad is not None:
                        print("GATModel", name, param.grad.norm().item())
            print("===================")

        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
        patch_transformer,
        multi_view,
        fusion,
        gat,
        loader,
        criterion,
        optimizer,
        device
):
    """

    :param patch_transformer: patch 级别的 transform模型
    :param multi_view: patch级别多视角模型
    :param fusion: trans 和 multi_view 融合模型
    :param gat: patch当作节点的图注意力模型
    :param loader: dataloader
    :param criterion: 计算损失
    :param optimizer: 参数更新
    :param device:
    :return:
    """
    patch_transformer.eval()
    multi_view.eval()
    fusion.eval()
    gat.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_trans = patch_transformer(x)
        x_view = multi_view(x)
        x_fused = fusion(x_trans, x_view)

        data_list = []
        for i in range(x_fused.size(0)):
            data = build_spherical_graph_coords(x_fused[i])
            data.y = y[i].unsqueeze(0)
            data_list.append(data)

        batch_garph = Batch.from_data_list(data_list).to(device)

        logits = gat(
            batch_garph.x,
            batch_garph.edge_index,
            batch_garph.batch
        )

        loss = criterion(logits, batch_garph.y)

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(current_dir, "Dataset")

    full_dataset = LungNoduleDataset(dataset_dir)
    num_samples = len(full_dataset)

    print("\n===== 五折交叉验证 =====")
    print(f"总样本数: {num_samples}")
    print("======================\n")

    # K-fold划分
    Kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(Kfold.split(range(num_samples))):
        print(f"\n===== Fold {fold + 1} / 5 =====")

        # 统计每折样本
        neg_tr, pos_tr = count_labels(full_dataset, train_idx)
        neg_val, pos_val = count_labels(full_dataset, val_idx)

        print(f"Train: {len(train_idx)} samples (neg={neg_tr}, pos={pos_tr})")
        print(f"Val: {len(val_idx)} samples (neg={neg_val}, pos={pos_val})")

        # 构建 subset
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        # 构建dataloader
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        # 测试迭代
        x, y = next(iter(train_loader))
        print(f"[DEBUG] 一个batch 大小: {x.shape}, {y}")

        # 创建模型
        trans_model = PatchTransformer().to(device)
        view_model = MultiViewFeature().to(device)
        fusion_model = FeatureFusion().to(device)
        gat_model = GATModel().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(trans_model.parameters()) +
            list(view_model.parameters()) +
            list(fusion_model.parameters()) +
            list(gat_model.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )

        best_val_acc = 0
        best_path = f"best_model_fold{fold+1}"
        save_dir = os.path.join(current_dir, "checkpoints", "GAT", best_path)
        os.makedirs(save_dir, exist_ok=True)
        print("保存路径:", save_dir)

        for epoch in range(30):
            train_loss, train_acc = train_one_epoch(
                patch_transformer=trans_model,
                multi_view=view_model,
                fusion=fusion_model,
                gat=gat_model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )

            val_loss, val_acc = validate(
                patch_transformer=trans_model,
                multi_view=view_model,
                fusion=fusion_model,
                gat=gat_model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )

            print(f"[Fold {fold + 1}] Epoch {epoch:02d}: "
                  f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
                  f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}")

            # 保存模型 TODO:可以用dict一键打包, 此处demo我们先逐个保存
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                torch.save(
                    trans_model.state_dict(),
                    os.path.join(save_dir, "trans_model.pth")
                )
                torch.save(
                    view_model.state_dict(),
                    os.path.join(save_dir, "view_model.pth")
                )
                torch.save(
                    fusion_model.state_dict(),
                    os.path.join(save_dir, "fusion_model.pth")
                )
                torch.save(
                    gat_model.state_dict(),
                    os.path.join(save_dir, "gat_model.pth")
                )
                print(f"保存成功")

        print(f"Fold {fold+1} Best Val Acc = {best_val_acc:.4f}")
        fold_results.append(best_val_acc)

    print("\n===== 五折结果 =====")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i + 1}: {acc:.4f}")
    print(f"平均准确率: {sum(fold_results) / 5:.4f}")
