import os
import gzip
import shutil
import SimpleITK as sitk
import os
import tqdm
import numpy as np
import glob
import re

paths = glob.glob(r"E:\RenYi\temp\*")

for p in tqdm.tqdm(paths):


    data = sitk.ReadImage(p)
    sp = p.split('\\')

    data_n = sitk.GetArrayFromImage(data)
    data_n[data_n < -1000] = -1000
    data_n[data_n > 200] = 200
    normalized_data = (data_n - np.min(data_n)) / (np.max(data_n) - np.min(data_n))
    normalized_data = sitk.GetImageFromArray(normalized_data)


    normalized_data.SetSpacing((1, 1, 1))
    normalized_data.SetOrigin((1,1,1))

    # sitk.WriteImage(norm_data, r"G:\Yikeda\nii_image\{}".format(sp[-1]).replace('.nii.gz','_0000.nii.gz'))
    sitk.WriteImage(normalized_data,
                    r"E:\RenYi\temp1/{}".format(
                        sp[-1]))


# nnUNetv2_predict -i G:\Yikeda\nii_image -o G:\Yikeda\nii_mask -d 002 -c 3d_fullres -f 0

