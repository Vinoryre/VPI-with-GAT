import SimpleITK as sitk
import os
from tqdm import tqdm
import glob
import numpy as np
# 定义DICOM文件夹路径和输出NIfTI文件路径
paths = glob.glob(r"I:\LungNolude_datasets\STAS\ZS_zhongshan\images\*\ICV")

for p in tqdm(paths):
    sp = p.split('\\')

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(p)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    data_n = sitk.GetArrayFromImage(image)
    normalized_data = (data_n - np.min(data_n)) / (np.max(data_n) - np.min(data_n))
    norm_data = sitk.GetImageFromArray(normalized_data)
    norm_data.SetSpacing((1,1,1))
    norm_data.SetOrigin((1,1,1))
    sitk.WriteImage(norm_data, r"I:\LungNolude_datasets\STAS\norm_data\ZS_zhongshan\IC\images\{}_{}.nii.gz".format(sp[-2],sp[-1]))









