"""
Turning angles files are produced with this script
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

new_datasets_list = INDIVIDUAL_DATASETS.copy()

for combined_dataset in [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]:
    new_datasets_list.append((combined_dataset, BTX_NOMENCLATURE))
    new_datasets_list.append((combined_dataset, CHOL_NOMENCLATURE))

for index, dataset in enumerate(new_datasets_list):
    print(dataset)
    filter_query = {'info.dataset': dataset, 'info.immobile': True} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1], 'info.immobile': True}

    with pd.ExcelWriter(f"./Results/{dataset}_{index}_gs_{APPLY_GS_CRITERIA}_angles_information.xlsx") as writer:
        all_angles = default_angles()
        
        for label in DIFFUSION_BEHAVIOURS_INFORMATION:
            label_angle_information = default_angles()

            ids = get_ids_of_trayectories_under_betha_limits(
                filter_query,
                DIFFUSION_BEHAVIOURS_INFORMATION[label]['range_0'],
                DIFFUSION_BEHAVIOURS_INFORMATION[label]['range_1'],
            )

            angles_infos = Trajectory._get_collection().find(
                {'_id': {'$in':[ObjectId(an_id) for an_id in ids]}},
                {f'info.analysis.angles_analysis':1}
            )

            for angle_info in angles_infos:
                for angle in angle_info['info']['analysis']['angles_analysis']:
                    label_angle_information[angle] += angle_info['info']['analysis']['angles_analysis'][angle]
                    all_angles[angle] += angle_info['info']['analysis']['angles_analysis'][angle]

            for angle in label_angle_information:
                frequency, bin_edges = custom_histogram(np.array(label_angle_information[angle]), 0, 180, 10)
                probability = frequency / np.sum(frequency)

                x_mid = []

                for i in range(len(probability)):
                    x_mid.append((bin_edges[i+1] + bin_edges[i]) / 2)

                delta_x = bin_edges[1] - bin_edges[0]

                f_x = probability / delta_x
                
                label_angle_information[angle] = f_x

            label_angle_information['x'] = x_mid

            new_angle_information = pd.DataFrame(label_angle_information)
            cols = new_angle_information.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            new_angle_information = new_angle_information[cols]
            new_angle_information.to_excel(writer, sheet_name=label, index=False)

        for angle in all_angles:
            frequency, bin_edges = custom_histogram(np.array(all_angles[angle]), 0, 180, 10)
            probability = frequency / np.sum(frequency)

            x_mid = []

            for i in range(len(probability)):
                x_mid.append((bin_edges[i+1] + bin_edges[i]) / 2)

            delta_x = bin_edges[1] - bin_edges[0]

            f_x = probability / delta_x
            
            all_angles[angle] = f_x

        all_angles['x'] = x_mid

        new_angle_information = pd.DataFrame(all_angles)
        cols = new_angle_information.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        new_angle_information = new_angle_information[cols]
        new_angle_information.to_excel(writer, sheet_name='all_angles', index=False)

        for label in ['confinement', 'non-confinement']:
            label_angle_information = default_angles()

            numeric_label = 1 if label == 'confinement' else 0

            angles_infos = Trajectory._get_collection().find(filter_query,{f'info.analysis.angles_by_state.{numeric_label}.angles':1})

            for angle_info in angles_infos:
                if 'analysis' in angle_info['info']:
                    for angle in angle_info['info']['analysis']['angles_by_state'][str(numeric_label)]['angles']:
                        label_angle_information[angle] += angle_info['info']['analysis']['angles_by_state'][str(numeric_label)]['angles'][angle]

            for angle in label_angle_information:
                frequency, bin_edges = custom_histogram(np.array(label_angle_information[angle]), 0, 180, 10)
                probability = frequency / np.sum(frequency)

                x_mid = []

                for i in range(len(probability)):
                    x_mid.append((bin_edges[i+1] + bin_edges[i]) / 2)

                delta_x = bin_edges[1] - bin_edges[0]

                f_x = probability / delta_x
                
                label_angle_information[angle] = f_x

            label_angle_information['x'] = x_mid

            new_angle_information = pd.DataFrame(label_angle_information)
            cols = new_angle_information.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            new_angle_information = new_angle_information[cols]
            new_angle_information.to_excel(writer, sheet_name=label, index=False)

DatabaseHandler.disconnect()
