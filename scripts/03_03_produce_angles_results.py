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

basic_info_file = open('./Results/basic_info.txt','w')

new_datasets_list = DATASETS_LIST.copy()[:-3]
new_datasets_list.append(BTX_NOMENCLATURE)
new_datasets_list.append(CHOL_NOMENCLATURE)

for index, dataset in enumerate(new_datasets_list):
    print(dataset)

    if dataset not in [BTX_NOMENCLATURE, CHOL_NOMENCLATURE]:
        filter_query = {'info.dataset': dataset, 'info.immobile': False} if APPLY_GS_CRITERIA else {'info.dataset': dataset}
    else:
        filter_query = {'$or': [{'info.classified_experimental_condition': dataset, 'info.immobile': False} if APPLY_GS_CRITERIA else {'info.classified_experimental_condition': dataset}]}

        if dataset == BTX_NOMENCLATURE:
            filter_query['$or'].append({'info.dataset': 'BTX680R', 'info.immobile': False} if APPLY_GS_CRITERIA else {'info.dataset': 'BTX680R'})
        else:
            filter_query['$or'].append({'info.dataset': 'CholesterolPEGKK114', 'info.immobile': False} if APPLY_GS_CRITERIA else {'info.dataset': 'CholesterolPEGKK114'})

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

basic_info_file.write(f"{BTX_NOMENCLATURE} -> n={Trajectory._get_collection().count_documents({'info.classified_experimental_condition':BTX_NOMENCLATURE})}\n")

fractions = []

for btx_id in Trajectory._get_collection().find({'info.classified_experimental_condition':BTX_NOMENCLATURE}, {f'id':1}):
    btx_trajectory = Trajectory.objects(id=btx_id['_id'])[0]
    if 'number_of_confinement_zones' in btx_trajectory.info and btx_trajectory.info['number_of_confinement_zones'] != 0:
        fractions.append(btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/btx_trajectory.info['number_of_confinement_zones'])

basic_info_file.write(f"{BTX_NOMENCLATURE} -> Fraction: {np.mean(fractions)}us, S.E.M: {sem(fractions)}s\n")

basic_info_file.write(f"{CHOL_NOMENCLATURE} -> n={Trajectory._get_collection().count_documents({'info.classified_experimental_condition':CHOL_NOMENCLATURE})}\n")

fractions = []

for chol_id in Trajectory._get_collection().find({'info.classified_experimental_condition':CHOL_NOMENCLATURE}, {f'id':1}):
    chol_trajectory = Trajectory.objects(id=chol_id['_id'])[0]
    if 'number_of_confinement_zones' in chol_trajectory.info and chol_trajectory.info['number_of_confinement_zones'] != 0:
        fractions.append(chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}']/chol_trajectory.info['number_of_confinement_zones'])

basic_info_file.write(f"{CHOL_NOMENCLATURE} -> Fraction: {np.mean(fractions)}us, S.E.M: {sem(fractions)}s\n")

basic_info_file.close()
DatabaseHandler.disconnect()
