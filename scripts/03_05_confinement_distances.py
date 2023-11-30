import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem

from CONSTANTS import *
from utils import *
from scipy.stats import expon


dataset_to_results = {
    'Control': ['Control_0_confinement_areas_distance.csv'],
    'CDx': ['CDx_1_confinement_areas_distance.csv'],
    'CholesterolPEGKK114': ['CholesterolPEGKK114_3_confinement_areas_distance.csv', 'fPEG-Chol_5_confinement_areas_distance.csv'],
    'BTX680R': ['BTX680R_2_confinement_areas_distance.csv', 'BTX680R_4_confinement_areas_distance.csv'],
}

for dataset in dataset_to_results:
    distances = []

    for file in dataset_to_results[dataset]:
        distances += pd.read_csv(os.path.join('Results',file))['confinement_areas_distance'].tolist()

    print(dataset, np.mean(distances), sem(distances))
