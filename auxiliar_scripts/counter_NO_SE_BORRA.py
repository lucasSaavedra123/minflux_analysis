from CONSTANTS import *
import pandas as pd
import glob
import numpy as np
from scipy.stats import sem

import matplotlib.pyplot as plt

"""

for file in glob.glob("./Results/*_gs_True_basic_information.xlsx"):
    print(file)
    r = pd.read_excel(file, sheet_name='betha')['betha'].tolist()
    counter = {}

    for d_label in DIFFUSION_BEHAVIOURS_INFORMATION:
        counter[d_label] = 0

        for angle in r:
            if DIFFUSION_BEHAVIOURS_INFORMATION[d_label]['range_0'] < angle < DIFFUSION_BEHAVIOURS_INFORMATION[d_label]['range_1']:
                counter[d_label] += 1
        
        #counter[d_label] /= len(r)
        #counter[d_label] *= 100

    print(counter)
"""

#Split Ks and Bethas

dataframe = {
    'file':[],
    'diffusive regime':[],
    'k mean':[],
    'k sem':[],
    'betha mean':[],
    'betha sem':[],
}

for file in glob.glob("./Results/*_gs_True_basic_information.xlsx"):
    print(file)
    b = np.array(pd.read_excel(file, sheet_name='betha')['betha'])
    k = np.array(pd.read_excel(file, sheet_name='k')['k'])
    #k = k[(b>0) & (b<2)]
    #b = b[(b>0) & (b<2)]

    for d_label in DIFFUSION_BEHAVIOURS_INFORMATION:
        boolean_flag = (b>DIFFUSION_BEHAVIOURS_INFORMATION[d_label]['range_0']) & (b<DIFFUSION_BEHAVIOURS_INFORMATION[d_label]['range_1'])
        dataframe['file'].append(file)
        dataframe['diffusive regime'].append(d_label)
        dataframe['k mean'].append(np.mean(k[boolean_flag]))
        dataframe['k sem'].append(sem(k[boolean_flag]))
        dataframe['betha mean'].append(np.mean(b[boolean_flag]))
        dataframe['betha sem'].append(sem(b[boolean_flag]))


pd.DataFrame(dataframe).to_csv('ks_bs.csv')