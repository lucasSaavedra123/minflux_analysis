import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon, sem
from scipy.spatial.distance import pdist
from bson.objectid import ObjectId

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *


APPLY_GS_CRITERIA = True

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
"""
#GET ALL RATIOS
pd.DataFrame({'ratio': get_list_of_values_of_field({'info.dataset': 'Control'}, 'ratio')}).to_csv('results/control_ratios.csv')
pd.DataFrame({'ratio': get_list_of_values_of_field({'info.dataset': 'CDx'}, 'ratio')}).to_csv('results/cdx_ratios.csv')

ratios = get_list_of_values_of_field({'info.dataset': 'BTX680R'}, 'ratio')
ratios += get_list_of_values_of_field({'info.classified_experimental_condition': BTX_NOMENCLATURE}, 'ratio')
pd.DataFrame({'ratio': ratios}).to_csv('results/btx_ratios.csv')

ratios = get_list_of_values_of_field({'info.dataset': 'CholesterolPEGKK114'}, 'ratio')
ratios += get_list_of_values_of_field({'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 'ratio')
pd.DataFrame({'ratio': ratios}).to_csv('results/chol_ratios.csv')
"""
"""
#GET ALL INTERVALS
list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'Control'}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
print(f'Control(n={len(list_of_trajectories_time)})-> Intervals', np.mean(intervals) * 1e6, sem(intervals) * 1e6)

list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'CDx'}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
print(f'CDx(n={len(list_of_trajectories_time)})->', np.mean(intervals) * 1e6, sem(intervals) * 1e6)

list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'BTX680R'}, 't') + get_list_of_main_field({'info.classified_experimental_condition': BTX_NOMENCLATURE}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
print(f'BTX(n={len(list_of_trajectories_time)})->', np.mean(intervals) * 1e6, sem(intervals) * 1e6)

list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'CholesterolPEGKK114'}, 't') + get_list_of_main_field({'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
print(f'Chol(n={len(list_of_trajectories_time)})->', np.mean(intervals) * 1e6, sem(intervals) * 1e6)
"""
DatabaseHandler.disconnect()