import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn .model_selection import KFold

from dataloader import LungNoduleDataset
from model import ThreeD_ResNet_18


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


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    训练单轮函数
    :param model:
    :param loader:
    :param criterion:
    :param optimizer:
    :param device:
    :return:
    """
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        _, pred = torch.max(outputs, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """
    验证函数
    :param model:
    :param loader:
    :param criterion:
    :param device:
    :return:
    """
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item() * x.size(0)

            _, pred = torch.max(outputs, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return running_loss / total, correct / total


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
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)

        # 测试迭代
        x, y = next(iter(train_loader))
        print(f"[DEBUG] 一个batch 大小: {x.shape}, {y}")

        # 创建模型
        model = ThreeD_ResNet_18(num_classes=2).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_val_acc = 0
        best_path = f"best_model_fold{fold+1}.pth"
        save_path = os.path.join(current_dir, "checkpoints", best_path)

        for epoch in range(30):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(f"[Fold {fold+1}] Epoch {epoch:02d}: "
                  f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
                  f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}")

            # 保存模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)

        print(f"Fold {fold+1} Best Val Acc = {best_val_acc:.4f}")
        fold_results.append(best_val_acc)

    print("\n===== 五折结果 =====")
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.4f}")
    print(f"平均准确率: {sum(fold_results)/5:.4f}")
