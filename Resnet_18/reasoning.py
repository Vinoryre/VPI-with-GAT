# 此脚本使用已经训练好的模型进行推理

import os
import torch
from torch.utils.data import DataLoader, Subset
from dataloader import LungNoduleDataset
from model import ThreeD_ResNet_18


def inference_on_patients(dataset_dir, model_path, patient_range=(30, 47), batch_size=1, device=None):
    """
    对指定编号的 positive 患者进行推理
    :param dataset_dir: Dataset 根目录
    :param model_path: 已训练模型路径
    :param patient_range: 只推理编号在 start-end 的患者, 如 (30, 47)
    :param batch_size: 推理批量大小
    :param device:
    :return:
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 1. 加载数据集
    full_dataset = LungNoduleDataset(dataset_dir)

    # 2. 找到 positive 患者索引
    positive_indices = []
    for idx, (path, label) in enumerate(full_dataset.samples):
        if label == 1:
            # path 形如 .../Patient_30_name/xxx
            patient_folder = os.path.basename(os.path.dirname(path))

            # 调试
            # print(f"[DEBUG] path={path}")
            # print(f"[DEBUG] patient_folder={patient_folder}")
            #
            # parts = patient_folder.split("_")
            # print(f"[DEBUG] parts after split: {parts}")

            patient_num = int(patient_folder.split("_")[1])
            if patient_range[0] <= patient_num <= patient_range[1]:
                positive_indices.append(idx)

    if len(positive_indices) == 0:
        print("[WARN] 没有找到符合条件的患者")
        return

    subset = Subset(full_dataset, positive_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    # 3. 创建模型, 加载权重
    model = ThreeD_ResNet_18(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    # 4. 推理循环
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1)
            for i in range(len(x)):
                results.append({
                    "patient_idx": positive_indices[i],
                    "true_label": int(y[i]),
                    "pred_label": int(pred[i]),
                    "pred_prob": probs[i].cpu().numpy()
                })

    # 5. 输出结果
    for r in results:
        print(f"PatientIdx={r['patient_idx']}, True={r['true_label']}, Pred={r['pred_label']}, Prob={r['pred_prob']}")

    return results


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(current_dir, "Dataset")
    model_path = os.path.join(current_dir, "checkpoints", "best_model_fold5.pth")

    results = inference_on_patients(dataset_dir, model_path, patient_range=(30, 47))

    if results:
        correct = sum(r["true_label"] == r["pred_label"] for r in results)
        total = len(results)
        accuracy = correct / total
        print(f"\n===== 推理统计 =====")
        print(f"样本数: {total}, 正确数: {correct}, 准确率: {accuracy:.4f}")
    else:
        print("没有推理结果")
