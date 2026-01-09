# 这个脚本将可视化 HU 截断 + Min-Max
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# 路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(current_dir, "Dataset")
save_path = os.path.join(current_dir, "Dataset", "positive", "Patient_1_libingyao", "first_patient.npy")

found = False
for label_name in ["negative", "positive"]:
    label_dir = os.path.join(dataset_dir, label_name)
    if not os.path.exists(label_dir):
        continue

    for patient in sorted(os.listdir(label_dir)):
        p_dir = os.path.join(label_dir, patient)
        if not os.path.isdir(p_dir):
            continue

        for f in os.listdir(p_dir):
            if f.endswith("_64_area.nii.gz"):
                nii_path = os.path.join(p_dir, f)
                print(f"[INFO] 处理患者: {patient}, 文件: {nii_path}")
                found = True
                break
        if found:
            break
    if found:
        break

if not found:
    raise FileNotFoundError("未找到任何 _64_area.nii.gz 文件!")

# 读取 nii
img = sitk.ReadImage(nii_path)
img_np = sitk.GetArrayFromImage(img).astype(np.float32)

# HU 截断
img_np = np.clip(img_np, -1000, 200)

# Min-Max 归一化
min_v = img_np.min()
max_v = img_np.max()
img_np_norm = (img_np - min_v) / (max_v - min_v + 1e-5)

# 保存为 npy
np.save(save_path, img_np_norm)
print(f"[SAVE] 保存处理后的图像至: {save_path}")

data = np.load(save_path)
plt.imshow(data[32])
plt.colorbar()
plt.show()
