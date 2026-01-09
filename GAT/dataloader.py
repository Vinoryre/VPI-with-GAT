import os
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import numpy as np


def extract_patches(img, patch_size=(32, 32, 32), stride=(32, 32, 32)):
    """
    3D ROI 分 patch
    :param img:
    :param patch_size:
    :param stride:
    :return:
    """
    C, D, H, W = img.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    patches = []
    for z in range(0, D - pd + 1, sd):
        for y in range(0, H - ph + 1, sh):
            for x in range(0, W - pw + 1, sw):
                patch = img[:, z:z+pd, y:y+ph, x:x+pw]
                patches.append(patch)

    patches = torch.stack(patches, dim=0)
    return patches


def flatten_patches(patches):
    """
    将 patch 展平为向量
    :param patches:
    :return:
    """
    N = patches.shape[0]
    return patches.view(N, -1)


class LungNoduleDataset(Dataset):
    def __init__(self, root_dir, patch_size=(8, 8, 8), stride=(8, 8, 8)):
        """
        自动遍历数据集,构建VPI阳性/阴性dataloader
        :param root_dir:
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.stride = stride

        # 存储 (patient_path, label)
        self.samples = []

        # 调试阴性,阳性数量
        self.pos_count = 0
        self.neg_count = 0

        for label_name, label_value in [("negative", 0), ("positive", 1)]:
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(class_dir):
                continue

            # 遍历患者文件夹
            for patient_name in os.listdir(class_dir):
                patient_path = os.path.join(class_dir, patient_name)

                if not os.path.isdir(patient_path):
                    continue

                # 查找 64^3 区域文件, 以 _64_area.nii.gz结尾
                cut_file = None
                for f in os.listdir(patient_path):
                    if f.endswith("_64_area.nii.gz"):
                        cut_file = os.path.join(patient_path, f)
                        break

                if cut_file is None:
                    print(f"[WARN] {patient_path} 找不到 *_64_area.nii.gz, 跳过")
                    continue

                self.samples.append((cut_file, label_value))

                # 统计逻辑
                if label_value == 1:
                    self.pos_count += 1
                else:
                    self.neg_count += 1

        print(f"[INFO] 总样本数: {len(self.samples)}")
        print(f"[INFO] 阳性: {self.pos_count}, 阴性: {self.neg_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nii_path, label = self.samples[idx]

        # 使用 SimpleITK 读取
        img = sitk.ReadImage(nii_path)
        img_np = sitk.GetArrayFromImage(img).astype(np.float32)
        assert isinstance(img_np, np.ndarray)

        # 数据预处理 HU截断 + Min-Max
        img_np = np.clip(img_np, -1000, 200)

        min_v = img_np.min()
        max_v = img_np.max()
        img_np = (img_np - min_v) / (max_v - min_v + 1e-5)

        # 转为 tensor
        # 增加通道
        img_np = np.expand_dims(img_np, 0)
        img_np = torch.from_numpy(img_np)

        # 切为patch
        patches = extract_patches(img_np, self.patch_size, self.stride)
        patches = flatten_patches(patches)

        return patches, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(current_dir, "Dataset")

    train_dataset = LungNoduleDataset(dataset_dir, patch_size=(8, 8, 8), stride=(8, 8, 8))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for x, y in train_loader:
        print(x.shape, y)
        break
