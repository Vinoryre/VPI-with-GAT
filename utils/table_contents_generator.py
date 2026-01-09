# 此脚本包含一些文件夹目录函数,用于精细控制项目目录格式

import os
import shutil
from glob import glob
from pypinyin import lazy_pinyin


def create_patient_dirs(root_dir, n_patients):
    """
    在 root_dir 下生成 n_patients 个患者目录, 每个患者包含 CT 和 lung_nodule_mask 子文件夹
    :param root_dir:
    :param n_patients:
    :return:
    """
    for i in range(1, n_patients + 1):
        patient_dir = os.path.join(root_dir, f"Patient_{i}_")
        ct_dir = os.path.join(patient_dir, "CT")
        mask_dir = os.path.join(patient_dir, "lung_nodule_mask")

        # 创建目录
        os.makedirs(ct_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        print(f"[INFO] 已创建: {patient_dir}")


def add_patients_to_dataset(dataset_dir, data_temp_dir, target_class='positive'):
    """
    将 Data_temp 下面的病人 DICOM 数据自动加入 Dataset/{positive, negative} 中,
    并且自动生成 Patient_x_pinyin 结构.
    :param dataset_dir: Dataset 根目录路径
    :param data_temp_dir: Data_temp 根目录路径
    :param target_class: 'positive' 或者 'negative'
    :return:
    """

    target_class = target_class.lower()
    assert target_class in ["positive", "negative"], "target_class 必须是 positive 或者 negative"

    target_path = os.path.join(dataset_dir, target_class)

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"{target_path} 不存在, 请检查 Dataset 目录结构")

    # 1). 统计已经有的 Patient_x_xxx 文件夹数目
    existing_patients = [
        d for d in os.listdir(target_path)
        if os.path.isdir(os.path.join(target_path, d)) and d.startswith("Patient_")
    ]
    next_index = len(existing_patients) + 1

    # 2). 遍历 Data_temp 下面的每一个病人文件夹
    for patient_folder in os.listdir(data_temp_dir):
        patient_path = os.path.join(data_temp_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        # 中文名 -> 拼音
        pinyin_name = "_".join(lazy_pinyin(patient_folder))

        # 新病人目录名
        new_patient_name = f"Patient_{next_index}_{pinyin_name}"
        new_patient_dir = os.path.join(target_path, new_patient_name)

        # 创建 CT, lung_nodule_mask 文件夹
        ct_dir = os.path.join(new_patient_dir, "CT")
        mask_dir = os.path.join(new_patient_dir, "lung_nodule_mask")

        os.makedirs(ct_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # 3. 递归寻找并移动所有 .dcm 文件到 CT/
        dcm_files = glob(os.path.join(patient_path, "**", "*.dcm"), recursive=True)

        if len(dcm_files) == 0:
            print(f" [WARNING] 病人 {patient_folder} 没有找到任何 .dcm 文件, 已跳过移动.")

        for dcm in dcm_files:
            shutil.move(dcm, ct_dir)

        print(f"已添加 {patient_folder} -> {new_patient_name}, 找到 {len(dcm_files)} 个 DICOM 文件")

        next_index += 1

    print("\n 所有病人已经成功添加并且重命名! ")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    positive_dir = os.path.join(current_dir, "Dataset", "positive")

    # n_patients = 30
    # create_patient_dirs

    dataset_dir = os.path.join(current_dir, "Dataset")
    data_temp_dir = os.path.join(current_dir, "Data_temp")

    add_patients_to_dataset(dataset_dir, data_temp_dir, target_class="positive")
