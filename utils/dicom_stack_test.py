# 此脚本用来测试DICOM本地与服务器顺序不同情况下能否复现
import os
import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk


def load_dicom_slices(dicom_dir, verbose=True):
    """
    读取并按照 DICOM 内部元数据排序
    :param dicom_dir:
    :param verbose:
    :return:
    """

    files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    files = [os.path.join(dicom_dir, f) for f in files]

    slices = []
    for f in files:
        try:
            d = pydicom.dcmread(f)
            slices.append(d)
        except:
            print(f"[WARNING] Failed to read {f}")

    if len(slices) == 0:
        raise RuntimeError("没有读取到 DICOM 文件, 请检查路径")

    # 优先使用 ImagePositionPatient 排序
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        print("Use ImagePositionPatient")
    except:
        # 回退到 InstanceNumber
        slices.sort(key=lambda x: int(x.InstanceNumber))
        print("Use InstanceNumber")

    # 打印每个切片信息
    if verbose:
        print("=== Slices info after sorting ===")
        for i, s in enumerate(slices):
            z = getattr(s, "ImagePositionPatient", [0,0,0])[2] if hasattr(s, "ImagePositionPatient") else "NA"
            print(f"{i:03d}: {os.path.basename(s.filename)}, z = {z}")

    return slices


def dicom_to_numpy(slices):
    """
    将 DICOM 切片堆叠成 numpy 3D volume + affine(LPS->RAS)
    :param slices:
    :return:
    """
    # 体数据 TODO: 加速实现
    volume = []
    for s in slices:
        img = s.pixel_array.astype(np.float32)
        if hasattr(s, 'RescaleSlope') and hasattr(s, 'RescaleIntercept'):
            img = img * float(s.RescaleSlope) + float(s.RescaleIntercept)
        volume.append(img)
    volume = np.stack(volume, axis=0)
    # 翻转 Z
    volume = np.flip(volume, axis=0)

    # 获取 DICOM 的空间信息
    s0 = slices[0]
    # 方向余弦
    # 前三个值为 row vector, 后三个值为 column vector
    iop = s0.ImageOrientationPatient
    row_cosine = np.array(iop[:3])
    col_cosine = np.array(iop[3:])
    slice_cosine = np.cross(row_cosine, col_cosine)

    # 像素间距
    spacing = list(map(float, s0.PixelSpacing))
    dz = float(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]) if len(slices) > 1 else 1.0

    # 构建 affine
    affine = np.eye(4)
    affine[:3, 0] = row_cosine * spacing[0]
    affine[:3, 1] = col_cosine * spacing[1]
    affine[:3, 2] = slice_cosine * dz
    affine[:3, 3] = slices[-1].ImagePositionPatient

    # Debug
    print("volume.shape:", volume.shape)
    print("affine:\n", affine)
    # affine修复
    affine_ras = affine.copy()
    affine_ras[:3, 0] *= -1
    affine_ras[:3, 1] *= -1
    affine_ras[:3, 2] *= -1
    affine_ras[:3, 3][0] *= -1
    affine_ras[:3, 3][1] *= -1
    print("affine_ras:\n", affine_ras)
    print("affine_ras[:3,2] (vz):", affine_ras[:3, 2])
    print("Nz, Ny, Nx:", volume.shape)
    print("expected origin if flip_z not applied:", affine_ras[:3, 3])
    print("expected origin if flip_z applied:", affine_ras[:3, 3] + affine_ras[:3, 2] * (volume.shape[0]-1))

    return volume, affine_ras


def dicom_to_nifti(dicom_dir, output_file):
    """
    单个 CT DICOM 文件夹转为 NIFTI
    :param dicom_dir:
    :param output_file:
    :return:
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # 转换为 NIFTI
    nifti_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
    nifti_image.SetSpacing(image.GetSpacing())
    nifti_image.SetOrigin(image.GetOrigin())

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sitk.WriteImage(nifti_image, output_file)
    print(f"已保存到: {output_file}")


def verify_reconstruction(orig, loaded):
    """
    对比两个体数据是否完全一致
    :param orig:
    :param loaded:
    :return:
    """
    if orig.shape != loaded.shape:
        print("形状不一致")
        return False

    equal = np.array_equal(orig, loaded)

    if equal:
        print("完全一致")
    else:
        print("有差异")

    return equal


if __name__ == "__main__":
    # 路径
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dicom_dir = os.path.join(current_dir, "Dataset", "positive", "Patient_1_libingyao", "CT")

    print("=== 开始读取并排序 ===")
    slices = load_dicom_slices(dicom_dir)

    print("=== 转换为numpy ===")
    volume, affine = dicom_to_numpy(slices)

    # 保存
    save_dir = os.path.join(current_dir, "Dataset", "positive", "Patient_1_libingyao")
    np.save(os.path.join(save_dir, "volume.npy"), volume)
    print("已保存为 volume.npy, shape = ", volume.shape)

    nii_img = nib.Nifti1Image(volume, affine)
    nib.save(nii_img, os.path.join(save_dir, "volume.nii.gz"))
    print("已保存为 volume.nii.gz")

    # 再次读取
    volume_loaded = np.load(os.path.join(save_dir, "volume.npy"))

    print("=== 验证 ===")
    verify_reconstruction(volume, volume_loaded)

    dicom_to_nifti(dicom_dir, os.path.join(save_dir, "sitk.nii.gz"))
