"""
Confinement distances are measured
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, kstest

from CONSTANTS import *
from utils import *


FONT_SIZE = 45
file = open('Results/confinement_distances.txt', 'w')
a = pd.read_csv(os.path.join('Results', 'Control_0_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
file.write('Control', np.mean(a), sem(a), '\n')

b = pd.read_csv(os.path.join('Results', 'CDx_1_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
file.write('CDx', np.mean(b), sem(b), '\n')

file.write('Control-CDx', kstest(a, b, alternative='two-sided'), '\n')

a = pd.read_csv(os.path.join('Results', 'CholesterolPEGKK114_3_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
file.write('fPEG-Chol', np.mean(a), sem(a), '\n')

b = pd.read_csv(os.path.join('Results', 'fPEG-Chol_5_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
file.write('fPEG-Chol with BTX', np.mean(b), sem(b), '\n')

file.write('fPEG-Chol', kstest(a, b, alternative='two-sided'), '\n')

a = pd.read_csv(os.path.join('Results', 'BTX680R_2_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
file.write('BTX', np.mean(a), sem(a), '\n')

b = pd.read_csv(os.path.join('Results', 'BTX680R_4_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
file.write('BTX with Chol', np.mean(b), sem(b), '\n')

file.write('BTX', kstest(a, b, alternative='two-sided'), '\n')

file.close()