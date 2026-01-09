# 此脚本的作用是将原生的.dcm 批量/single 转换为肺结节部分周边的 128^3 ROI区域
# TODO: 裁剪出来的是128^3, 但是文件名字隐性地是64^3, 后期需要进行修复
import SimpleITK as sitk
import os
import numpy as np
import traceback
from scipy import ndimage


def read_dicom_series(dicom_dir):
    """
    读取 .dcm 返回itk_image类型
    :param dicom_dir:
    :return:
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def sitk_to_numpy(image):
    """
    读取itk_image类型,返回 numpy
    :param image:
    :return:
    """
    return sitk.GetArrayFromImage(image)


def get_nodule_center(nodule_mask_path):
    """
    获取肺结节质点, (zyx坐标numpy, xyz坐标切片, 世界坐标)
    如果是多个连通区域的话, 则只选取连通域最大的中心
    同时使用 bounding box 算法来定位肿瘤球
    :param nodule_mask_path:
    :return:
    """
    # 读取掩码
    mask = sitk.ReadImage(nodule_mask_path)

    # 转为 numpy
    mask_arr = sitk.GetArrayFromImage(mask)

    # 1).连通域标记
    labeled, num = ndimage.label(mask_arr > 0)
    if num == 0:
        raise ValueError("掩码中没有结节.")

    # 2).找到最大连通块
    sizes = ndimage.sum(mask_arr > 0, labeled, range(1, num + 1))
    max_label = np.argmax(sizes) + 1

    main_component = (labeled == max_label)

    # 3).计算最大结节的 bounding box
    coords = np.argwhere(main_component)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    # bounding box 中心
    zyx_center = np.array([(zmin + zmax) / 2,
                           (ymin + ymax) / 2,
                           (xmin + xmax) / 2], dtype=np.float32)

    # zyx_index->physical 坐标
    def index_to_world_float(mask, index_float_zyx):
        spacing = np.array(mask.GetSpacing())
        origin = np.array(mask.GetOrigin())
        direction = np.array(mask.GetDirection())

        # 构造为 3x3 矩阵
        direction = direction.reshape(3, 3)

        # float index 转为 xyz
        index_xyz = np.array(index_float_zyx[::-1])

        # world = origin + direction @ (index * spacing)
        world_center = origin + direction.dot(index_xyz * spacing)

        return index_xyz, world_center

    xyz_center, phys_center = index_to_world_float(mask, zyx_center)

    return zyx_center, xyz_center, phys_center


def world_to_index(image, world_point):
    """
    世界坐标转换为 index 坐标
    :param image:
    :param world_point:
    :return:
    """
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    # index_xyz = inv(R) @ ((world - origin) / spacing)
    inv_dir = np.linalg.inv(direction)
    index_xyz = inv_dir.dot(world_point - origin) / spacing

    # ITK(x,y,z) -> numpy(z,y,x)
    return index_xyz[::-1]


def crop_patch(img_np, center_zyx, size=128):
    """
    裁剪出 128^3 的区域
    :param img_np:
    :param center_zyx:
    :param size:
    :return:
    """
    half = size // 2
    zc, yc, xc = center_zyx

    z_min = int(zc - half)
    z_max = z_min + size
    y_min = int(yc - half)
    y_max = y_min + size
    x_min = int(xc - half)
    x_max = x_min + size

    # boundary handling
    z_min = max(z_min, 0)
    y_min = max(y_min, 0)
    x_min = max(x_min, 0)

    z_max = min(z_max, img_np.shape[0])
    y_max = min(y_max, img_np.shape[1])
    x_max = min(x_max, img_np.shape[2])

    patch = img_np[z_min:z_max, y_min:y_max, x_min:x_max]

    # 浮点数可能会导致patch的尺寸超64,所以先裁剪
    patch = patch[:size, :size, :size]

    # 不够 64 ,使用 0 padding
    pad_z = size - patch.shape[0]
    pad_y = size - patch.shape[1]
    pad_x = size - patch.shape[2]

    patch = np.pad(patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

    return patch


def numpy_to_sitk(patch_np, ref_image, center_zyx):
    """
    将裁剪后的 numpy patch 转换为 sitk_image
    :param patch_np:
    :param ref_image:
    :param center_zyx:
    :return:
    """
    img = sitk.GetImageFromArray(patch_np.astype(np.float32))

    # 保留 spacing/direction
    img.SetSpacing(ref_image.GetSpacing())
    img.SetDirection(ref_image.GetDirection())

    # 调整 origin
    spacing = np.array(ref_image.GetSpacing())
    direction = np.array(ref_image.GetDirection()).reshape(3, 3)
    origin = np.array(ref_image.GetOrigin())

    # patch 左前上角的 index
    corner_zyx = center_zyx - np.array([patch_np.shape[0], patch_np.shape[1], patch_np.shape[2]])/2
    corner_xyz = corner_zyx[::-1]

    # origin_new = world_of(corner_xyz)
    corner_world = origin + direction.dot(corner_xyz * spacing)

    img.SetOrigin(tuple(corner_world))

    return img


def save_patch_nifti(img, output_path):
    """
    保存为 nifti 格式
    :param img:
    :param output_path:
    :return:
    """
    sitk.WriteImage(img, output_path)
    print(f"Saved patch to {output_path}")


def extract_64_patch(dicom_dir, nodule_mask_path, output_nii):
    """
    裁剪 64^3 区域的自动化管道
    :param dicom_dir:
    :param nodule_mask_path:
    :param output_nii:
    :return:
    """
    # 读取 ct 图像
    ct_img = read_dicom_series(dicom_dir)

    # 获取肺结节的世界坐标
    _, _, nodule_phys_center = get_nodule_center(nodule_mask_path)

    # 肺结节的世界坐标转换为在ct图像的index坐标
    ct_index_zyx = world_to_index(ct_img, nodule_phys_center)

    # ct 的 itk_image->numpy
    ct_np = sitk.GetArrayFromImage(ct_img)

    # 开始裁剪
    patch_np = crop_patch(ct_np, ct_index_zyx)

    # 裁剪出来的 numpy -> sitk_image
    patch_img = numpy_to_sitk(patch_np, ct_img, ct_index_zyx)

    # 保存为 .nii.gz
    save_patch_nifti(patch_img, output_nii)


def batch_process_dataset(dataset_dir):
    """
    遍历 dataset_dir 下面的 positive/negative 文件夹,对每个患者调用extract_64_patch, 然后保存为 Patient_x_64_area.nii.gz
    :param dataset_dir:
    :return:
    """
    for label in ["positive", "negative"]:
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.exists(label_dir):
            print(f"[WARN] 文件夹不存在: {label_dir}, 跳过")
            continue

        for patient_name in sorted(os.listdir(label_dir)):
            patient_dir = os.path.join(label_dir, patient_name)
            if not os.path.isdir(patient_dir):
                continue

            ct_dir = os.path.join(patient_dir, "CT")
            mask_dir = os.path.join(patient_dir, "lung_nodule_mask")

            # 查找 lung_nodule_mask 里面的第一个 .nii.gz文件
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")]
            if len(mask_files) == 0:
                print(f"[WARN] 没有找到掩码文件: {mask_files}, 跳过患者 {patient_name}")
                continue
            nodule_mask_path = os.path.join(mask_dir, mask_files[0])

            # 输出路径, 提取 Patient_x
            base_name = "_".join(patient_name.split("_")[:2])
            output_path = os.path.join(patient_dir, f"{base_name}_64_area.nii.gz")

            try:
                print(f"[INFO] 处理 {patient_name} ...")
                extract_64_patch(ct_dir, nodule_mask_path, output_path)
            except Exception as e:
                print(f"[ERROR] {patient_name} 处理失败: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dicom_dir = os.path.join(current_dir, "Dataset", "positive", "Patient_1_libingyao", "CT")
    dataset_dir = os.path.join(current_dir, "Dataset")

    # image = read_dicom_series(dicom_dir)
    # print(f"spacing: {image.GetSpacing()}")
    # print(f"origin: {image.GetOrigin()}")
    # print(f"direction: {image.GetDirection()}")
    #
    # img_nparr = sitk_to_numpy(image)
    # print(f"img_nparr: {img_nparr}\n "
    #       f"img_nparr.shape: {img_nparr.shape}")
    #
    # nodule_mask_path = os.path.join(current_dir, "Dataset", "positive", "Patient_1_libingyao", "lung_nodule_mask", "mask1_libingyao.nii.gz")
    #
    # nodule_zyx_center, nodule_xyz_center, nodule_world_center = get_nodule_center(nodule_mask_path)
    #
    # print(f"zyx_center: {nodule_zyx_center}\n"
    #       f"xyz_center: {nodule_xyz_center}\n"
    #       f"world_center: {nodule_world_center}\n")
    #
    # Patient_1_64_area_save_path= os.path.join(current_dir, "Dataset", "positive", "Patient_1_libingyao", "Patient_1_64_area.nii.gz")
    #
    # extract_64_patch(dicom_dir, nodule_mask_path, Patient_1_64_area_save_path)

    batch_process_dataset(dataset_dir)
