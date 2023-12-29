"""
From the paper, this is the "basic" characterization
of trajectories.
"""
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
file = open('Results/preliminary_results.txt', 'w')

file.write("PERCENTAGE OF IMMOBILIZED TRAJECTORIES\n")
quantity = Trajectory._get_collection().count_documents({'info.dataset':DATASETS_LIST[0]})
immobile_quantity = Trajectory._get_collection().count_documents({'info.dataset':DATASETS_LIST[0], 'info.immobile':True})
file.write(DATASETS_LIST[0], immobile_quantity/quantity, "\n")

quantity = Trajectory._get_collection().count_documents({'info.dataset':DATASETS_LIST[1]})
immobile_quantity = Trajectory._get_collection().count_documents({'info.dataset':DATASETS_LIST[1], 'info.immobile':True})
file.write(DATASETS_LIST[1], immobile_quantity/quantity, "\n")

quantity = Trajectory._get_collection().count_documents({'info.classified_experimental_condition':BTX_NOMENCLATURE})
immobile_quantity = Trajectory._get_collection().count_documents({'info.classified_experimental_condition':BTX_NOMENCLATURE, 'info.immobile':True})
file.write(BTX_NOMENCLATURE, immobile_quantity/quantity, "\n")

quantity = Trajectory._get_collection().count_documents({'info.dataset':DATASETS_LIST[2]})
immobile_quantity = Trajectory._get_collection().count_documents({'info.dataset':DATASETS_LIST[2], 'info.immobile':True})
file.write(DATASETS_LIST[2], immobile_quantity/quantity, "\n")

quantity = Trajectory._get_collection().count_documents({'info.classified_experimental_condition':CHOL_NOMENCLATURE})
immobile_quantity = Trajectory._get_collection().count_documents({'info.classified_experimental_condition':CHOL_NOMENCLATURE, 'info.immobile':True})
file.write(CHOL_NOMENCLATURE, immobile_quantity/quantity, "\n")

quantity = Trajectory._get_collection().count_documents({'info.dataset':DATASETS_LIST[3]})
immobile_quantity = Trajectory._get_collection().count_documents({'info.dataset':DATASETS_LIST[3], 'info.immobile':True})
file.write(DATASETS_LIST[3], immobile_quantity/quantity, "\n")

file.write("RATIOS\n")
pd.DataFrame({'ratio': get_list_of_values_of_field({'info.dataset': 'Control'}, 'ratio')}).to_csv('Results/control_ratios.csv')
pd.DataFrame({'ratio': get_list_of_values_of_field({'info.dataset': 'CDx'}, 'ratio')}).to_csv('Results/cdx_ratios.csv')

ratios = get_list_of_values_of_field({'info.dataset': 'BTX680R'}, 'ratio')
ratios += get_list_of_values_of_field({'info.classified_experimental_condition': BTX_NOMENCLATURE}, 'ratio')
pd.DataFrame({'ratio': ratios}).to_csv('Results/btx_ratios.csv')

ratios = get_list_of_values_of_field({'info.dataset': 'CholesterolPEGKK114'}, 'ratio')
ratios += get_list_of_values_of_field({'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 'ratio')
pd.DataFrame({'ratio': ratios}).to_csv('Results/chol_ratios.csv')

file.write("ALL INTERVALS\n")
list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'Control'}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'Control(n={len(list_of_trajectories_time)})-> Intervals', np.mean(intervals) * 1e6, sem(intervals) * 1e6, "\n")

list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'CDx'}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'CDx(n={len(list_of_trajectories_time)})->', np.mean(intervals) * 1e6, sem(intervals) * 1e6, "\n")

list_of_trajectories_time = get_list_of_main_field({'info.classified_experimental_condition': BTX_NOMENCLATURE}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'{BTX_NOMENCLATURE} with Chol (n={len(list_of_trajectories_time)})->', np.mean(intervals) * 1e6, sem(intervals) * 1e6, "\n")

list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'BTX680R'}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'BTX680R(n={len(list_of_trajectories_time)})->', np.mean(intervals) * 1e6, sem(intervals) * 1e6, "\n")

list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'CholesterolPEGKK114'}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'CholesterolPEGKK114(n={len(list_of_trajectories_time)})->', np.mean(intervals) * 1e6, sem(intervals) * 1e6, "\n")

list_of_trajectories_time = get_list_of_main_field({'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 't')
list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'{CHOL_NOMENCLATURE}(n={len(list_of_trajectories_time)})->', np.mean(intervals) * 1e6, sem(intervals) * 1e6, "\n")

file.write("ALL DISTANCE INTERVALS\n")
list_of_trajectories_positions = get_list_of_main_field({'info.dataset': 'Control'}, 'x') + get_list_of_main_field({'info.dataset': 'Control'}, 'y')
list_of_trajectories_positions = [l for l in list_of_trajectories_positions if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.abs(np.diff(position_list)) for position_list in list_of_trajectories_positions]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'Control(n={len(list_of_trajectories_positions)})-> (nm) Distance Intervals {np.mean(intervals) * 1e3} {np.std(intervals) * 1e3} {sem(intervals) * 1e3}\n')

list_of_trajectories_positions = get_list_of_main_field({'info.dataset': 'CDx'}, 'x') + get_list_of_main_field({'info.dataset': 'Control'}, 'y')
list_of_trajectories_positions = [l for l in list_of_trajectories_positions if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.abs(np.diff(position_list)) for position_list in list_of_trajectories_positions]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'CDx(n={len(list_of_trajectories_positions)})-> (nm) Distance Intervals {np.mean(intervals) * 1e3} {np.std(intervals) * 1e3} {sem(intervals) * 1e3}\n')

list_of_trajectories_positions = get_list_of_main_field({'info.classified_experimental_condition': BTX_NOMENCLATURE}, 'x') + get_list_of_main_field({'info.classified_experimental_condition': BTX_NOMENCLATURE}, 'y')
list_of_trajectories_positions = [l for l in list_of_trajectories_positions if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.abs(np.diff(position_list)) for position_list in list_of_trajectories_positions]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'{BTX_NOMENCLATURE} with Chol (n={len(list_of_trajectories_positions)})-> (nm) Distance Intervals {np.mean(intervals) * 1e3} {np.std(intervals) * 1e3} {sem(intervals) * 1e3}\n')

list_of_trajectories_positions = get_list_of_main_field({'info.dataset': 'BTX680R'}, 'x') + get_list_of_main_field({'info.dataset': 'BTX680R'}, 'y')
list_of_trajectories_positions = [l for l in list_of_trajectories_positions if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.abs(np.diff(position_list)) for position_list in list_of_trajectories_positions]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'BTX680R(n={len(list_of_trajectories_positions)})-> (nm) Distance Intervals {np.mean(intervals) * 1e3} {np.std(intervals) * 1e3} {sem(intervals) * 1e3}\n')

list_of_trajectories_positions = get_list_of_main_field({'info.dataset': 'CholesterolPEGKK114'}, 'x') + get_list_of_main_field({'info.dataset': 'CholesterolPEGKK114'}, 'y')
list_of_trajectories_positions = [l for l in list_of_trajectories_positions if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.abs(np.diff(position_list)) for position_list in list_of_trajectories_positions]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'CholesterolPEGKK114(n={len(list_of_trajectories_positions)})-> (nm) Distance Intervals {np.mean(intervals) * 1e3} {np.std(intervals) * 1e3} {sem(intervals) * 1e3}\n')

list_of_trajectories_positions = get_list_of_main_field({'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 'x') + get_list_of_main_field({'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 'y')
list_of_trajectories_positions = [l for l in list_of_trajectories_positions if len(l) > 1]
intervals = list(itertools.chain.from_iterable([np.abs(np.diff(position_list)) for position_list in list_of_trajectories_positions]))
intervals = [interval for interval in intervals if interval != 0]
file.write(f'{CHOL_NOMENCLATURE}(n={len(list_of_trajectories_positions)})-> (nm) Distance Intervals {np.mean(intervals) * 1e3} {np.std(intervals) * 1e3} {sem(intervals) * 1e3}\n')

file.close()
DatabaseHandler.disconnect()