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
import tqdm
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *


INDIVIDUAL_DATASETS = [
    'Control',
    'CDx',
    'BTX680R',
    'CholesterolPEGKK114',
    'CK666-BTX680',
    'CK666-CHOL',
    'BTX640-CHOL-50-nM',
    'BTX640-CHOL-50-nM-LOW-DENSITY',
]

COMBINED_DATASETS = [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)
file = open('./Results/preliminary_results.txt', 'w')

file.write("PERCENTAGE OF IMMOBILIZED TRAJECTORIES\n")
for dataset in tqdm.tqdm(INDIVIDUAL_DATASETS):
    quantity = Trajectory._get_collection().count_documents({'info.dataset':dataset})
    immobile_quantity = Trajectory._get_collection().count_documents({'info.dataset':dataset, 'info.immobile':True})
    file.write(f'{dataset} {immobile_quantity/quantity}\n')

for dataset in tqdm.tqdm(COMBINED_DATASETS):
    quantity = Trajectory._get_collection().count_documents({'info.dataset': dataset, 'info.classified_experimental_condition': BTX_NOMENCLATURE})
    immobile_quantity = Trajectory._get_collection().count_documents({'info.dataset':dataset, 'info.classified_experimental_condition': BTX_NOMENCLATURE, 'info.immobile':True})
    file.write(f'({BTX_NOMENCLATURE}) {dataset} {immobile_quantity/quantity}\n')

    quantity = Trajectory._get_collection().count_documents({'info.dataset': dataset, 'info.classified_experimental_condition': CHOL_NOMENCLATURE})
    immobile_quantity = Trajectory._get_collection().count_documents({'info.dataset':dataset, 'info.classified_experimental_condition': CHOL_NOMENCLATURE, 'info.immobile':True})
    file.write(f'({CHOL_NOMENCLATURE}) {dataset} {immobile_quantity/quantity}\n')

file.write("RATIOS\n")
for dataset in tqdm.tqdm(INDIVIDUAL_DATASETS):
    pd.DataFrame({'ratio': get_list_of_values_of_field({'info.dataset': dataset}, 'ratio')}).to_csv(f'Results/{dataset}_ratios.csv')

for dataset in tqdm.tqdm(COMBINED_DATASETS):
    pd.DataFrame({'ratio': get_list_of_values_of_field({'info.dataset': dataset, 'info.classified_experimental_condition': BTX_NOMENCLATURE}, 'ratio')}).to_csv(f'Results/{dataset}_{BTX_NOMENCLATURE}_ratios.csv')
    pd.DataFrame({'ratio': get_list_of_values_of_field({'info.dataset': dataset, 'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 'ratio')}).to_csv(f'Results/{dataset}_{CHOL_NOMENCLATURE}_ratios.csv')

file.write("ALL INTERVALS\n")
for dataset in tqdm.tqdm(INDIVIDUAL_DATASETS):
    list_of_trajectories_time = get_list_of_main_field({'info.dataset': dataset}, 't')
    list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
    intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
    intervals = [interval for interval in intervals if interval != 0]
    file.write(f'{dataset}(n={len(list_of_trajectories_time)})-> Intervals {np.mean(intervals) * 1e6} {sem(intervals) * 1e6}\n')

for dataset in tqdm.tqdm(COMBINED_DATASETS):
    list_of_trajectories_time = get_list_of_main_field({'info.dataset': dataset, 'info.classified_experimental_condition': BTX_NOMENCLATURE}, 't')
    list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
    intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
    intervals = [interval for interval in intervals if interval != 0]
    file.write(f'{BTX_NOMENCLATURE} ({dataset}) (n={len(list_of_trajectories_time)})-> Intervals {np.mean(intervals) * 1e6} {sem(intervals) * 1e6}\n')

    list_of_trajectories_time = get_list_of_main_field({'info.dataset': dataset, 'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 't')
    list_of_trajectories_time = [l for l in list_of_trajectories_time if len(l) > 1]
    intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
    intervals = [interval for interval in intervals if interval != 0]
    file.write(f'{CHOL_NOMENCLATURE} ({dataset}) (n={len(list_of_trajectories_time)})-> Intervals {np.mean(intervals) * 1e6} {sem(intervals) * 1e6}\n')

file.write("ALL DISTANCE INTERVALS\n")
for dataset in tqdm.tqdm(INDIVIDUAL_DATASETS):
    list_of_trajectories_positions = get_list_of_positions({'info.dataset': dataset})
    list_of_trajectories_positions = [np.power(np.diff(X), 2) for X in list_of_trajectories_positions]
    list_of_trajectories_positions = [np.sqrt(X[0,:] + X[1,:]) for X in list_of_trajectories_positions]
    intervals = list(itertools.chain.from_iterable(list_of_trajectories_positions))
    intervals = [interval for interval in intervals if interval != 0]
    file.write(f'{dataset} (n={len(list_of_trajectories_positions)})-> (nm) Distance Intervals {np.mean(intervals) * 1e3} {np.std(intervals) * 1e3} {sem(intervals) * 1e3}\n')

for dataset in tqdm.tqdm(COMBINED_DATASETS):
    list_of_trajectories_positions = get_list_of_positions({'info.dataset': dataset, 'info.classified_experimental_condition': BTX_NOMENCLATURE})
    list_of_trajectories_positions = [np.power(np.diff(X), 2) for X in list_of_trajectories_positions]
    list_of_trajectories_positions = [np.sqrt(X[0,:] + X[1,:]) for X in list_of_trajectories_positions]
    intervals = list(itertools.chain.from_iterable(list_of_trajectories_positions))
    intervals = [interval for interval in intervals if interval != 0]
    file.write(f'{BTX_NOMENCLATURE} ({dataset}) (n={len(list_of_trajectories_positions)})-> (nm) Distance Intervals {np.mean(intervals) * 1e3} {np.std(intervals) * 1e3} {sem(intervals) * 1e3}\n')

    list_of_trajectories_positions = get_list_of_positions({'info.dataset': dataset, 'info.classified_experimental_condition': CHOL_NOMENCLATURE})
    list_of_trajectories_positions = [np.power(np.diff(X), 2) for X in list_of_trajectories_positions]
    list_of_trajectories_positions = [np.sqrt(X[0,:] + X[1,:]) for X in list_of_trajectories_positions]
    intervals = list(itertools.chain.from_iterable(list_of_trajectories_positions))
    intervals = [interval for interval in intervals if interval != 0]
    file.write(f'{CHOL_NOMENCLATURE} ({dataset}) (n={len(list_of_trajectories_positions)})-> (nm) Distance Intervals {np.mean(intervals) * 1e3} {np.std(intervals) * 1e3} {sem(intervals) * 1e3}\n')

file.write('LENGTHS\n')
for dataset in tqdm.tqdm(INDIVIDUAL_DATASETS):
    list_of_trajectories_time = get_list_of_main_field({'info.dataset': dataset}, 't')
    lengths = [len(time_list) for time_list in list_of_trajectories_time]
    lengths = [length for length in lengths if length > 1]
    file.write(f'{dataset}(n={len(lengths)})-> Intervals {np.mean(lengths)} {sem(lengths)}\n')

for dataset in tqdm.tqdm(COMBINED_DATASETS):
    list_of_trajectories_time = get_list_of_main_field({'info.dataset': dataset, 'info.classified_experimental_condition': BTX_NOMENCLATURE}, 't')
    lengths = [len(time_list) for time_list in list_of_trajectories_time]
    lengths = [length for length in lengths if length > 1]
    file.write(f'{BTX_NOMENCLATURE} ({dataset}) (n={len(lengths)})-> Intervals {np.mean(lengths)} {sem(lengths)}\n')

    list_of_trajectories_time = get_list_of_main_field({'info.dataset': dataset, 'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 't')
    lengths = [len(time_list) for time_list in list_of_trajectories_time]
    lengths = [length for length in lengths if length > 1]
    file.write(f'{CHOL_NOMENCLATURE} ({dataset}) (n={len(lengths)})-> Intervals {np.mean(lengths)} {sem(lengths)}\n')

file.write('DURATION\n')
for dataset in tqdm.tqdm(INDIVIDUAL_DATASETS):
    list_of_trajectories_time = get_list_of_main_field({'info.dataset': dataset}, 't')
    durations = [time_list[-1] - time_list[0] for time_list in list_of_trajectories_time]
    durations = [duration for duration in durations if duration != 0]
    file.write(f'{dataset}(n={len(durations)})-> Intervals {np.mean(durations)} {sem(durations)}\n')

for dataset in tqdm.tqdm(COMBINED_DATASETS):
    list_of_trajectories_time = get_list_of_main_field({'info.dataset': dataset, 'info.classified_experimental_condition': BTX_NOMENCLATURE}, 't')
    durations = [time_list[-1] - time_list[0] for time_list in list_of_trajectories_time]
    durations = [duration for duration in durations if duration != 0]
    file.write(f'{BTX_NOMENCLATURE} ({dataset}) (n={len(durations)})-> Intervals {np.mean(durations)} {sem(durations)}\n')

    list_of_trajectories_time = get_list_of_main_field({'info.dataset': dataset, 'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 't')
    durations = [time_list[-1] - time_list[0] for time_list in list_of_trajectories_time]
    durations = [duration for duration in durations if duration != 0]
    file.write(f'{CHOL_NOMENCLATURE} ({dataset}) (n={len(durations)})-> Intervals {np.mean(durations)} {sem(durations)}\n')

file.close()
DatabaseHandler.disconnect()