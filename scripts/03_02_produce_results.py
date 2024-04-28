"""
All important results like areas, axis lengths, etc. 
are produced within this file.
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
    print(dataset,index)

    basic_query_dict = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1]}
    with pd.ExcelWriter(f"./Results/{dataset}_{index}_gs_{APPLY_GS_CRITERIA}_basic_information.xlsx") as writer:
        #Data that take into account GS criteria
        filter_query = basic_query_dict.copy()
        filter_query.update({'info.immobile': False} if APPLY_GS_CRITERIA else {})

        pd.DataFrame({'k': get_list_of_values_of_analysis_field(filter_query, 'k')}).to_excel(writer, sheet_name='k', index=False)
        pd.DataFrame({'betha': get_list_of_values_of_analysis_field(filter_query, 'betha')}).to_excel(writer, sheet_name='betha', index=False)
        #pd.DataFrame({'dc': get_list_of_values_of_analysis_field(filter_query, 'directional_coefficient')}).to_excel(writer, sheet_name='directional_coefficient', index=False)
        pd.DataFrame({'d_2_4': get_list_of_values_of_analysis_field(filter_query, 'd_2_4')}).to_excel(writer, sheet_name='d_2_4', index=False)
        pd.DataFrame({'localization_precision': get_list_of_values_of_analysis_field(filter_query, 'localization_precision')}).to_excel(writer, sheet_name='localization_precision', index=False)
        pd.DataFrame({'meanDP': get_list_of_values_of_analysis_field(filter_query, 'meanDP')}).to_excel(writer, sheet_name='meanDP', index=False)
        pd.DataFrame({'corrDP': get_list_of_values_of_analysis_field(filter_query, 'corrDP')}).to_excel(writer, sheet_name='corrDP', index=False)
        pd.DataFrame({'AvgSignD': get_list_of_values_of_analysis_field(filter_query, 'AvgSignD')}).to_excel(writer, sheet_name='AvgSignD', index=False)

        residence_times = get_list_of_values_of_analysis_field(filter_query, 'residence_time')
        residence_times = [residence_time for residence_time in residence_times if residence_time != 0]
        pd.DataFrame({'residence_time': residence_times}).to_excel(writer, sheet_name='residence_time', index=False)

        inverse_residence_times = get_list_of_values_of_analysis_field(filter_query, 'inverse_residence_time')
        inverse_residence_times = [inverse_residence_time for inverse_residence_time in inverse_residence_times if inverse_residence_time != 0]
        pd.DataFrame({'inverse_residence_time': inverse_residence_times}).to_excel(writer, sheet_name='inverse_residence_time', index=False)

        def measure_ratio(residence, inverse):
            if residence is None or inverse is None:
                return None
            else:
                return residence/(inverse+residence)

        residence_times_and_durations = [document for document in Trajectory._get_collection().find(filter_query, {f'info.analysis.inverse_residence_time':1, f'info.analysis.residence_time': 1})]
        residence_times_and_durations = [measure_ratio(document.get('info', {}).get('analysis', {}).get('residence_time'), document.get('info', {}).get('analysis', {}).get('inverse_residence_time')) for document in residence_times_and_durations]
        residence_times_and_durations = [value for value in residence_times_and_durations if value is not None]
        pd.DataFrame({'residence_times_and_durations': residence_times_and_durations}).to_excel(writer, sheet_name='residence_ratios', index=False)

        def measure_rate(state_array, time_array):
            if state_array is None or time_array is None:
                return None
            else:
                return np.abs(np.diff(state_array)!=0).sum() / (time_array[-1] - time_array[0])

        confinement_rates = [document for document in Trajectory._get_collection().find(filter_query, {f'info.analysis.confinement-states':1, f't': 1})]
        confinement_rates = [measure_rate(document.get('info', {}).get('analysis', {}).get('confinement-states'), document.get('t')) for document in confinement_rates]
        confinement_rates = [value for value in confinement_rates if value is not None]
        pd.DataFrame({'change_rates': confinement_rates}).to_excel(writer, sheet_name='change_rates', index=False)

        list_of_semi_major_axis = get_list_of_values_of_analysis_field(filter_query, 'confinement-a')
        list_of_semi_major_axis = list(itertools.chain.from_iterable(list_of_semi_major_axis))
        pd.DataFrame({'semi_major_axis': list_of_semi_major_axis}).to_excel(writer, sheet_name='semi_major_axis', index=False)

        browian_portions = []
        confined_portions = []

        for result in Trajectory._get_collection().find(filter_query, {'_id':1}):
            a_trajectory = Trajectory.objects(id=str(result['_id']))[0]
            sub_trajectories_by_state = a_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=False)

            for sub_trajectory in sub_trajectories_by_state[0]:
                browian_portions.append(sub_trajectory.duration)

            for sub_trajectory in sub_trajectories_by_state[1]:
                confined_portions.append(sub_trajectory.duration)

        pd.DataFrame({'browian_portions': [p for p in browian_portions if p != 0]}).to_excel(writer, sheet_name='browian_portions', index=False)
        pd.DataFrame({'confined_portions': [p for p in confined_portions if p != 0]}).to_excel(writer, sheet_name='confined_portions', index=False)
        """
        #Data that takes all trajectories
        filter_query = basic_query_dict.copy()
        pd.DataFrame({'ratio': get_list_of_values_of_field(filter_query, 'ratio')}).to_excel(writer, sheet_name='ratio', index=False)

        list_of_trajectories_time = get_list_of_main_field(filter_query, 't')

        lengths = [len(time_list) for time_list in list_of_trajectories_time]
        lengths = [length for length in lengths if length != 1]
        pd.DataFrame({'length': lengths}).to_excel(writer, sheet_name='length', index=False)

        durations = [time_list[-1] - time_list[0] for time_list in list_of_trajectories_time]
        durations = [duration for duration in durations if duration != 0]
        pd.DataFrame({'duration': durations}).to_excel(writer, sheet_name='duration', index=False)
        """
        #Data that takes all mobile trajectories
        filter_query = basic_query_dict.copy()
        filter_query.update({'info.immobile': False})

        list_of_number_of_trajectories_per_overlap = get_list_of_values_of_analysis_field(filter_query, 'number_of_trajectories_per_overlap')
        list_of_number_of_trajectories_per_overlap = list(itertools.chain.from_iterable(list_of_number_of_trajectories_per_overlap))
        pd.DataFrame({'number_of_trajectories_per_overlap': list_of_number_of_trajectories_per_overlap}).to_excel(writer, sheet_name='number_of_trajectories_per_overlap', index=False)

        list_of_semi_major_axis = get_list_of_values_of_analysis_field(filter_query, 'confinement-a')
        list_of_semi_major_axis = list(itertools.chain.from_iterable(list_of_semi_major_axis))
        pd.DataFrame({'semi_major_axis': list_of_semi_major_axis}).to_excel(writer, sheet_name='semi_major_axis', index=False)

        list_of_eccentricities = get_list_of_values_of_analysis_field(filter_query, 'confinement-e')
        list_of_eccentricities = list(itertools.chain.from_iterable(list_of_eccentricities))
        pd.DataFrame({'eccentricity': list_of_eccentricities}).to_excel(writer, sheet_name='eccentricity', index=False)

        list_of_confinement_areas = get_list_of_values_of_analysis_field(filter_query, 'confinement-area')
        list_of_confinement_areas = list(itertools.chain.from_iterable([[area * 1e6 for area in areas] for areas in list_of_confinement_areas]))
        pd.DataFrame({'area': list_of_confinement_areas}).to_excel(writer, sheet_name='area', index=False)

        list_of_steps = get_list_of_values_of_analysis_field(filter_query, 'non-confinement-steps')
        list_of_steps = list(itertools.chain.from_iterable(list_of_steps))
        pd.DataFrame({'non-confinement-steps': list_of_steps}).to_excel(writer, sheet_name='non-confinement-steps', index=False)

        list_of_steps = get_list_of_values_of_analysis_field(filter_query, 'confinement-steps')
        list_of_steps = list(itertools.chain.from_iterable(list_of_steps))
        pd.DataFrame({'confinement-steps': list_of_steps}).to_excel(writer, sheet_name='confinement-steps', index=False)

        list_of_bethas = get_list_of_values_of_analysis_field(filter_query, 'non-confinement-betha')
        list_of_bethas = list(itertools.chain.from_iterable(list_of_bethas))
        pd.DataFrame({'non-confinement-betha': list_of_bethas}).to_excel(writer, sheet_name='non-confinement-betha', index=False)

        list_of_bethas = get_list_of_values_of_analysis_field(filter_query, 'confinement-betha')
        list_of_bethas = list(itertools.chain.from_iterable(list_of_bethas))
        pd.DataFrame({'confinement-betha': list_of_bethas}).to_excel(writer, sheet_name='confinement-betha', index=False)

        list_of_ks = get_list_of_values_of_analysis_field(filter_query, 'non-confinement-k')
        list_of_ks = list(itertools.chain.from_iterable(list_of_ks))
        pd.DataFrame({'non-confinement-k': list_of_ks}).to_excel(writer, sheet_name='non-confinement-k', index=False)

        list_of_ks = get_list_of_values_of_analysis_field(filter_query, 'confinement-k')
        list_of_ks = list(itertools.chain.from_iterable(list_of_ks))
        pd.DataFrame({'confinement-k': list_of_ks}).to_excel(writer, sheet_name='confinement-k', index=False)

        list_of_ds = get_list_of_values_of_analysis_field(filter_query, 'non-confinement-d_2_4')
        list_of_ds = list(itertools.chain.from_iterable(list_of_ds))
        pd.DataFrame({'non-confinement-d_2_4': list_of_ds}).to_excel(writer, sheet_name='non-confinement-d_2_4', index=False)

        list_of_ds = get_list_of_values_of_analysis_field(filter_query, 'confinement-d_2_4')
        list_of_ds = list(itertools.chain.from_iterable(list_of_ds))
        pd.DataFrame({'confinement-d_2_4': list_of_ds}).to_excel(writer, sheet_name='confinement-d_2_4', index=False)

        list_of_confinement_areas_centroids = get_list_of_values_of_analysis_field(filter_query, 'confinement_areas_centroids')
        list_of_confinement_areas_centroids = list(itertools.chain.from_iterable([pdist(np.array(confinement_areas_centroids) * 1e3).tolist() for confinement_areas_centroids in list_of_confinement_areas_centroids if len(confinement_areas_centroids) >= 2]))
        pd.DataFrame({'confinement_areas_distance': list_of_confinement_areas_centroids}).to_csv(f'./Results/{dataset}_{index}_confinement_areas_distance.csv')

basic_info_file = open('./Results/basic_info.txt','w')

for combined_dataset in [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]:
    fractions = []
    for btx_id in Trajectory._get_collection().find({'info.dataset': combined_dataset, 'info.classified_experimental_condition':BTX_NOMENCLATURE, 'info.immobile': False}, {f'id':1}):
        btx_trajectory = Trajectory.objects(id=btx_id['_id'])[0]
        if 'number_of_confinement_zones' in btx_trajectory.info and btx_trajectory.info['number_of_confinement_zones'] != 0:
            fractions.append(btx_trajectory.info[f'number_of_confinement_zones_with_{CHOL_NOMENCLATURE}']/btx_trajectory.info['number_of_confinement_zones'])

    np.savetxt(f'./Results/{combined_dataset}-{BTX_NOMENCLATURE}_fraction.txt', fractions, fmt='%f')
    basic_info_file.write(f"{combined_dataset}-{BTX_NOMENCLATURE} Fraction: {np.mean(fractions)}, S.E.M: {sem(fractions)}s\n")

    fractions = []
    for chol_id in Trajectory._get_collection().find({'info.dataset': combined_dataset, 'info.classified_experimental_condition':CHOL_NOMENCLATURE, 'info.immobile': False}, {f'id':1}):
        chol_trajectory = Trajectory.objects(id=chol_id['_id'])[0]
        if 'number_of_confinement_zones' in chol_trajectory.info and chol_trajectory.info['number_of_confinement_zones'] != 0:
            fractions.append(chol_trajectory.info[f'number_of_confinement_zones_with_{BTX_NOMENCLATURE}']/chol_trajectory.info['number_of_confinement_zones'])
    
    np.savetxt(f'./Results/{combined_dataset}-{CHOL_NOMENCLATURE}_fraction.txt', fractions, fmt='%f')
    basic_info_file.write(f"{combined_dataset}-{CHOL_NOMENCLATURE} Fraction: {np.mean(fractions)}, S.E.M: {sem(fractions)}s\n")

basic_info_file.close()

overlap_non_confinement_portion_info = open('./Results/overlap_non_confinement_portion_info.txt','w')

for combined_dataset in [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]:
    fractions = []
    for btx_id in Trajectory._get_collection().find({'info.dataset': combined_dataset, 'info.classified_experimental_condition':BTX_NOMENCLATURE, 'info.immobile': False}, {f'id':1}):
        btx_trajectory = Trajectory.objects(id=btx_id['_id'])[0]
        
        if f'{CHOL_NOMENCLATURE}_single_intersections' in btx_trajectory.info:
            for non_confined_portion in btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[0]:
                fractions.append(np.sum(non_confined_portion.info[f'{CHOL_NOMENCLATURE}_single_intersections'])/non_confined_portion.length)

    np.savetxt(f'./Results/{combined_dataset}-{BTX_NOMENCLATURE}_non_confinement_overlap_fraction.txt', fractions, fmt='%f')
    overlap_non_confinement_portion_info.write(f"{combined_dataset}-{BTX_NOMENCLATURE} Fraction: {np.mean(fractions)}, S.E.M: {sem(fractions)}s\n")

    fractions = []
    for chol_id in Trajectory._get_collection().find({'info.dataset': combined_dataset, 'info.classified_experimental_condition':CHOL_NOMENCLATURE, 'info.immobile': False}, {f'id':1}):
        chol_trajectory = Trajectory.objects(id=chol_id['_id'])[0]

        if f'{BTX_NOMENCLATURE}_single_intersections' in btx_trajectory.info:
            for non_confined_portion in btx_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, use_info=True)[0]:
                fractions.append(np.sum(non_confined_portion.info[f'{BTX_NOMENCLATURE}_single_intersections'])/non_confined_portion.length)
    
    np.savetxt(f'./Results/{combined_dataset}-{CHOL_NOMENCLATURE}_non_confinement_overlap_fraction.txt', fractions, fmt='%f')
    overlap_non_confinement_portion_info.write(f"{combined_dataset}-{CHOL_NOMENCLATURE} Fraction: {np.mean(fractions)}, S.E.M: {sem(fractions)}s\n")

overlap_non_confinement_portion_info.close()

DatabaseHandler.disconnect()
