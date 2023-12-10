import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, kstest

from CONSTANTS import *
from utils import *


a = pd.read_csv(os.path.join('Results', 'Control_0_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
print('Control', np.mean(a), sem(a))

b = pd.read_csv(os.path.join('Results', 'CDx_1_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
print('CDx', np.mean(b), sem(b))

print('Control-CDx', kstest(a, b, alternative='two-sided'))

plt.hist(a, density=True, bins='auto', histtype='stepfilled', alpha=0.5, color='red')
plt.hist(b, density=True, bins='auto', histtype='stepfilled', alpha=0.5, color='blue')
plt.xlim(0)
plt.xlabel('Confinement sojourns distance [nm]', fontsize=30)
plt.ylabel('Frequency', fontsize=30)
plt.yticks([])
plt.xticks(fontsize=30)
plt.show()

a = pd.read_csv(os.path.join('Results', 'CholesterolPEGKK114_3_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
print('fPEG-Chol', np.mean(a), sem(a))

b = pd.read_csv(os.path.join('Results', 'fPEG-Chol_5_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
print('fPEG-Chol with BTX', np.mean(b), sem(b))

print('fPEG-Chol', kstest(a, b, alternative='two-sided'))

plt.hist(a, density=True, bins='auto', histtype='stepfilled', alpha=0.5, color='orange')
plt.hist(b, density=True, bins='auto', histtype='stepfilled', alpha=0.5, color='black')
plt.xlim(0)
plt.xlabel('Confinement sojourns distance [nm]', fontsize=30)
plt.ylabel('Frequency', fontsize=30)
plt.yticks([])
plt.xticks(fontsize=30)
plt.show()

a = pd.read_csv(os.path.join('Results', 'BTX680R_2_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
print('BTX', np.mean(a), sem(a))

b = pd.read_csv(os.path.join('Results', 'BTX680R_4_confinement_areas_distance.csv'))['confinement_areas_distance'].tolist()
print('BTX with Chol', np.mean(b), sem(b))

print('BTX', kstest(a, b, alternative='two-sided'))

plt.hist(a, density=True, bins='auto', histtype='stepfilled', alpha=0.5, color='green')
plt.hist(b, density=True, bins='auto', histtype='stepfilled', alpha=0.5, color='pink')
plt.xlim(0)
plt.xlabel('Confinement sojourns distance [nm]', fontsize=30)
plt.ylabel('Frequency', fontsize=30)
plt.yticks([])
plt.xticks(fontsize=30)
plt.show()