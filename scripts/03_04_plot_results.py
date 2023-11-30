import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from CONSTANTS import *
from utils import *
from scipy.stats import expon


dataset_to_results = {
    'Control': ['Control_0_basic_information.xlsx'],
    'CDx': ['CDx_1_basic_information.xlsx'],
    'CholesterolPEGKK114': ['CholesterolPEGKK114_3_basic_information.xlsx', 'fPEG-Chol_5_basic_information.xlsx'],
    'BTX680R': ['BTX680R_2_basic_information.xlsx', 'BTX680R_4_basic_information.xlsx'],
}

for dataset in dataset_to_results:
    residence_times = []

    for file in dataset_to_results[dataset]:
        residence_times += pd.read_excel(open(os.path.join('Results',file), 'rb'), sheet_name='residence_time')['residence_time'].tolist()

    #Exponential Fitting
    loc, scale = expon.fit(residence_times, floc=0)

    x = np.arange(0.01,4,0.001)
    pdfs = expon.pdf(x, loc=0, scale=scale)

    plt.plot(x,pdfs, color=DATASET_TO_COLOR[dataset])
    plt.hist(residence_times, density=True, bins='auto', histtype='stepfilled', alpha=0.2, color=DATASET_TO_COLOR[dataset])

plt.xlabel('Confinement Time [s]', fontname="Arial", fontsize=30)
plt.yticks(visible=False)
plt.xlim([0,2])
plt.ylim([0,3])
plt.xticks([0,1,2])
plt.xticks(fontname="Arial", fontsize=30)
plt.ylabel('Frequency', fontname="Arial", fontsize=30)
plt.tight_layout()
plt.show()
