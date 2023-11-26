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
list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'Control'}, 't')
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
print('Control->', np.mean(intervals))

list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'CDx'}, 't')
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
print('CDx->', np.mean(intervals))

list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'BTX680R'}, 't') + get_list_of_main_field({'info.classified_experimental_condition': BTX_NOMENCLATURE}, 't')
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
print('BTX->', np.mean(intervals))

list_of_trajectories_time = get_list_of_main_field({'info.dataset': 'CholesterolPEGKK114'}, 't') + get_list_of_main_field({'info.classified_experimental_condition': CHOL_NOMENCLATURE}, 't')
intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
intervals = [interval for interval in intervals if interval != 0]
print('Chol->', np.mean(intervals))
"""

basic_info_file = open('./Results/basic_info.txt','w')

new_datasets_list = DATASETS_LIST.copy()
new_datasets_list = DATASETS_LIST[:-1]
new_datasets_list.append(BTX_NOMENCLATURE)
new_datasets_list.append(CHOL_NOMENCLATURE)

for dataset in new_datasets_list:
    SEARCH_FIELD = 'info.dataset' if dataset not in [BTX_NOMENCLATURE, CHOL_NOMENCLATURE] else 'info.classified_experimental_condition'
    with pd.ExcelWriter(f"./Results/{dataset}_basic_information.xlsx") as writer:
        #Data that take into account GS criteria
        filter_query = {SEARCH_FIELD: dataset, 'info.immobile': False} if APPLY_GS_CRITERIA else {SEARCH_FIELD: dataset}

        pd.DataFrame({'k': get_list_of_values_of_analysis_field(filter_query, 'k')}).to_excel(writer, sheet_name='k', index=False)
        pd.DataFrame({'betha': get_list_of_values_of_analysis_field(filter_query, 'betha')}).to_excel(writer, sheet_name='betha', index=False)
        pd.DataFrame({'dc': get_list_of_values_of_analysis_field(filter_query, 'directional_coefficient')}).to_excel(writer, sheet_name='directional_coefficient', index=False)

        residence_times = get_list_of_values_of_analysis_field(filter_query, 'residence_time')
        residence_times = [residence_time for residence_time in residence_times if residence_time != 0]
        pd.DataFrame({'residence_time': residence_times}).to_excel(writer, sheet_name='residence_time', index=False)

        #Data that takes all trajectories
        filter_query = {SEARCH_FIELD: dataset}
        pd.DataFrame({'ratio': get_list_of_values_of_field(filter_query, 'ratio')}).to_excel(writer, sheet_name='ratio', index=False)

        list_of_trajectories_time = get_list_of_main_field(filter_query, 't')
        intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
        intervals = [interval for interval in intervals if interval != 0]
        #pd.DataFrame({'interval': intervals}).to_excel(writer, sheet_name='interval', index=False)

        lengths = [len(time_list) for time_list in list_of_trajectories_time]
        lengths = [length for length in lengths if length != 1]
        pd.DataFrame({'length': lengths}).to_excel(writer, sheet_name='length', index=False)

        durations = [time_list[-1] - time_list[0] for time_list in list_of_trajectories_time]
        durations = [duration for duration in durations if duration != 0]
        pd.DataFrame({'duration': durations}).to_excel(writer, sheet_name='duration', index=False)

        #Data that takes all mobile trajectories
        filter_query = {SEARCH_FIELD: dataset, 'info.immobile': False}

        list_of_semi_major_axis = get_list_of_values_of_analysis_field(filter_query, 'confinement-a')
        list_of_semi_major_axis = list(itertools.chain.from_iterable([semi_major_axis for semi_major_axis in list_of_semi_major_axis]))
        pd.DataFrame({'semi_major_axis': list_of_semi_major_axis}).to_excel(writer, sheet_name='semi_major_axis', index=False)

        list_of_eccentricities = get_list_of_values_of_analysis_field(filter_query, 'confinement-e')
        list_of_eccentricities = list(itertools.chain.from_iterable([eccentricities for eccentricities in list_of_eccentricities]))
        pd.DataFrame({'eccentricity': list_of_eccentricities}).to_excel(writer, sheet_name='eccentricity', index=False)

        list_of_confinement_areas = get_list_of_values_of_analysis_field(filter_query, 'confinement-area')
        list_of_confinement_areas = list(itertools.chain.from_iterable([[area * 1e6 for area in areas] for areas in list_of_confinement_areas]))
        pd.DataFrame({'area': list_of_confinement_areas}).to_excel(writer, sheet_name='area', index=False)

        basic_info_file.write(f"{dataset} -> n={len(lengths)}\n")
        basic_info_file.write(f"{dataset} -> Time Interval Mean: {np.mean(np.array(intervals)*1e6)}us, S.E.M: {sem(np.array(intervals)*1e6)}us\n")
        basic_info_file.write(f"{dataset} -> Residence Time Scale: {np.mean(residence_times)}us, S.E.M: {sem(residence_times)}s\n")

        list_of_confinement_areas_centroids = get_list_of_values_of_analysis_field(filter_query, 'confinement_areas_centroids')
        list_of_confinement_areas_centroids = list(itertools.chain.from_iterable([pdist(np.array(confinement_areas_centroids) * 1e3).tolist() for confinement_areas_centroids in list_of_confinement_areas_centroids if len(confinement_areas_centroids) >= 2]))
        pd.DataFrame({'confinement_areas_distance': list_of_confinement_areas_centroids}).to_csv(f'./Results/{dataset}_confinement_areas_distance.csv')

    with pd.ExcelWriter(f"./Results/{dataset}_angles_information.xlsx") as writer:
        all_angles = default_angles()
        
        for label in DIFFUSION_BEHAVIOURS_INFORMATION:
            label_angle_information = default_angles()

            filter_query = {SEARCH_FIELD: dataset, 'info.immobile': False} if APPLY_GS_CRITERIA else {SEARCH_FIELD: dataset}

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

            filter_query = {SEARCH_FIELD: dataset, 'info.immobile': False} if APPLY_GS_CRITERIA else {SEARCH_FIELD: dataset}

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

    #Exponential Fitting
    loc, scale = expon.fit(residence_times, floc=0)

    x = np.arange(0.01,4,0.001)
    pdfs = expon.pdf(x, loc=0, scale=scale)

    if dataset not in [CHOL_NOMENCLATURE, BTX_NOMENCLATURE]:
        plt.plot(x,pdfs, color=DATASET_TO_COLOR[dataset])
        plt.hist(residence_times, density=True, bins='auto', histtype='stepfilled', alpha=0.2, color=DATASET_TO_COLOR[dataset])

basic_info_file.write(f"{BTX_NOMENCLATURE} -> n={Trajectory._get_collection().count_documents({'info.classified_experimental_condition':BTX_NOMENCLATURE, 'info.immobile':False})}\n")

fractions = []

for btx_id in Trajectory._get_collection().find({'info.classified_experimental_condition':BTX_NOMENCLATURE, 'info.immobile':False}, {f'id':1}):
    btx_trajectory = Trajectory.objects(id=btx_id['_id'])[0]
    if 'number_of_confinement_zones' in btx_trajectory.info and btx_trajectory.info['number_of_confinement_zones'] != 0:
        fractions.append(btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/btx_trajectory.info['number_of_confinement_zones'])

basic_info_file.write(f"{BTX_NOMENCLATURE} -> Fraction: {np.mean(fractions)}us, S.E.M: {sem(fractions)}s\n")

basic_info_file.write(f"{CHOL_NOMENCLATURE} -> n={Trajectory._get_collection().count_documents({'info.classified_experimental_condition':CHOL_NOMENCLATURE, 'info.immobile':False})}\n")

fractions = []

for chol_id in Trajectory._get_collection().find({'info.classified_experimental_condition':CHOL_NOMENCLATURE, 'info.immobile':False}, {f'id':1}):
    chol_trajectory = Trajectory.objects(id=chol_id['_id'])[0]
    if 'number_of_confinement_zones' in chol_trajectory.info and chol_trajectory.info['number_of_confinement_zones'] != 0:
        fractions.append(chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}']/chol_trajectory.info['number_of_confinement_zones'])

basic_info_file.write(f"{CHOL_NOMENCLATURE} -> Fraction: {np.mean(fractions)}us, S.E.M: {sem(fractions)}s\n")

basic_info_file.close()
DatabaseHandler.disconnect()

plt.xlabel('Confinement Time [s]', fontname="Arial", fontsize=40)
plt.yticks(visible=False)
plt.xlim([0,4])
plt.xticks(fontname="Arial", fontsize=40)
plt.ylabel('Frequency', fontname="Arial", fontsize=40)

plt.show()
