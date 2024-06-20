"""
Upon intersection measure, we measure which
BTX confinement zones overlap with those
of Chol confinement.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from CONSTANTS import *
from utils import extract_dataset_file_roi_file

CHOL_AND_BTX_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

file_and_rois = [info for info in extract_dataset_file_roi_file() if info[0] in CHOL_AND_BTX_DATASETS]
passed_rois_counter = 0
for dataset, file, roi in file_and_rois:
    if os.path.exists(f'./overlaps_significant_test_files/{dataset}_{file}_{roi}.txt'):
        ious = np.loadtxt(f'./overlaps_significant_test_files/{dataset}_{file}_{roi}.txt')
        real_iou = ious[-1]
        ious = ious[:-1]

        q_9500 = np.quantile(ious, 0.9500)

        #plt.hist(ious)
        #plt.axvline(real_iou, color='black')
        #plt.axvline(q_9500, color='red')
        #plt.tight_layout()
        #plt.show()

        if q_9500 < real_iou:
            passed_rois_counter += 1

print(passed_rois_counter/len(file_and_rois))