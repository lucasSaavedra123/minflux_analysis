from CONSTANTS import *
import pandas as pd




r = pd.read_excel('./Results/fPEG-Chol_5_gs_True_basic_information.xlsx', sheet_name='non-confinement-betha')['non-confinement-betha'].tolist()
counter = {}

for d_label in DIFFUSION_BEHAVIOURS_INFORMATION:
    counter[d_label] = 0

    for angle in r:
        if DIFFUSION_BEHAVIOURS_INFORMATION[d_label]['range_0'] < angle < DIFFUSION_BEHAVIOURS_INFORMATION[d_label]['range_1']:
            counter[d_label] += 1
    
    counter[d_label] /= len(r)
    counter[d_label] *= 100

print(counter)