import glob

from CONSTANTS import *
import pandas as pd
import glob
import numpy as np
from scipy.stats import sem

dataset = {
    'file': [],
    '%': [],
    'result': []
}

for file in glob.glob("./Results/*_gs_True_number_of_trajectories_per_overlap.xlsx"):
    counters = np.array(pd.read_excel(file, sheet_name='number_of_trajectories_per_overlap')['number_of_trajectories_per_overlap'])
    dataset['file'].append(file)
    dataset['%'].append(len(counters[counters==0])/len(counters))
    
    counters = counters[counters!=0]

    dataset['result'].append(f'{np.mean(counters)} ± {sem(counters)}')

pd.DataFrame(dataset).to_csv('r.csv')