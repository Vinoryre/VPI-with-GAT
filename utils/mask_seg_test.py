# 脚本查看肺结节数据哪个是人工标注,哪个是模型推理
import os
import nibabel as nib
import numpy as np

# 根目录
dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
mask_dir = os.path.join(dataset_root, 'positive', 'mask')
seg_dir = os.path.join(dataset_root, 'positive', 'segmentation')


def analyze_folder(folder_path):
    print(f"\nAnalyzing folder: {folder_path}")
    files = [f for f in os.listdir(folder_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
    for f in files:
        file_path = os.path.join(folder_path, f)
        img = nib.load(file_path).get_fdata()
        unique_vals = np.unique(img)
        min_val, max_val = img.min(), img.max()
        nonzero_count = np.count_nonzero(img)
        total_voxels = img.size
        print(f"File: {f}")
        print(f"    Unique values: {unique_vals}")
        print(f"    Min/Max: {min_val}/{max_val}")
        print(f"    Non-zero voxels: {nonzero_count} / {total_voxels} ({100*nonzero_count/total_voxels:.2f}%)\n")


if __name__ == "__main__":
    analyze_folder(mask_dir)
    analyze_folder(seg_dir)
