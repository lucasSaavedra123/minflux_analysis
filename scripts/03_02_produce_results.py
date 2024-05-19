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


def reduce_mean_by_file_and_roi(tuples):
    values_per_roi = defaultdict(lambda: [])
    for value, file, roi in tuples:
        if value is not None:
            values_per_roi[file+roi].append(value)

    for alias in values_per_roi:
        values_per_roi[alias] = np.mean(values_per_roi[alias])   
    list_of_values = list(values_per_roi.values())
    return list_of_values

def unpack_data(list_of_data, mean_by_roi):
    if not mean_by_roi:
        list_of_data = list(itertools.chain.from_iterable(list_of_data))
    return list_of_data

def upload_data_to_writer(writer, filter_query, field, mean_by_roi, unpack_data_neccesary, filter_condition=None, multiplier=None):
    list_of_data = get_list_of_values_of_analysis_field(filter_query, field, mean_by_roi=mean_by_roi)
    if unpack_data_neccesary and not mean_by_roi:
        list_of_data = unpack_data(list_of_data, mean_by_roi)
    
    if filter_condition is not None:
        list_of_data = [value for value in list_of_data if filter_condition(value)]

    if multiplier is not None:
        list_of_data = [value*multiplier for value in list_of_data]

    pd.DataFrame({field: list_of_data}).to_excel(writer, sheet_name=f'{field}_{mean_by_roi}', index=False)

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
        for mean_by_roi in [False, True]:
            #Data that take into account GS criteria
            filter_query = basic_query_dict.copy()
            filter_query.update({'info.immobile': False} if APPLY_GS_CRITERIA else {})

            for false_unpack_fields in ['k', 'betha', 'd_2_4', 'localization_precision', 'meanDP', 'corrDP', 'AvgSignD']:
                upload_data_to_writer(writer, filter_query, false_unpack_fields, mean_by_roi, False)

            upload_data_to_writer(writer, filter_query, 'residence_time', mean_by_roi, False, filter_condition=lambda a: a!=0)
            upload_data_to_writer(writer, filter_query, 'inverse_residence_time', mean_by_roi, False, filter_condition=lambda a: a!=0)

            def measure_ratio(residence, inverse):
                if residence is None or inverse is None:
                    return None
                else:
                    return residence/(inverse+residence)

            if not mean_by_roi:
                residence_times_and_durations = [document for document in Trajectory._get_collection().find(filter_query, {f'info.analysis.inverse_residence_time':1, f'info.analysis.residence_time': 1, 'info.file':1,'info.roi':1})]
                residence_times_and_durations = [measure_ratio(document.get('info', {}).get('analysis', {}).get('residence_time'), document.get('info', {}).get('analysis', {}).get('inverse_residence_time')) for document in residence_times_and_durations]
            else:
                residence_times_and_durations = [document for document in Trajectory._get_collection().find(filter_query, {f'info.analysis.inverse_residence_time':1, f'info.analysis.residence_time': 1, 'info.file':1,'info.roi':1})]
                residence_times_and_durations = [(measure_ratio(document.get('info', {}).get('analysis', {}).get('residence_time'), document.get('info', {}).get('analysis', {}).get('inverse_residence_time')), document.get('info').get('file'), document.get('info').get('roi')) for document in residence_times_and_durations]
                residence_times_and_durations = reduce_mean_by_file_and_roi(residence_times_and_durations)

            residence_times_and_durations = [value for value in residence_times_and_durations if value is not None]

            pd.DataFrame({'residence_times_and_durations': residence_times_and_durations}).to_excel(writer, sheet_name=f'residence_ratios_{mean_by_roi}', index=False)

            def measure_rate(state_array, time_array):
                if state_array is None or time_array is None:
                    return None
                else:
                    return np.abs(np.diff(state_array)!=0).sum() / (time_array[-1] - time_array[0])

            if not mean_by_roi:
                confinement_rates = [document for document in Trajectory._get_collection().find(filter_query, {f'info.analysis.confinement-states':1, f't': 1})]
                confinement_rates = [measure_rate(document.get('info', {}).get('analysis', {}).get('confinement-states'), document.get('t')) for document in confinement_rates]
            else:
                confinement_rates = [document for document in Trajectory._get_collection().find(filter_query, {f'info.analysis.confinement-states':1, f't': 1, 'info.file':1,'info.roi':1})]
                confinement_rates = [(measure_rate(document.get('info', {}).get('analysis', {}).get('confinement-states'), document.get('t')), document.get('info').get('file'), document.get('info').get('roi')) for document in confinement_rates]
                confinement_rates = reduce_mean_by_file_and_roi(confinement_rates)

            confinement_rates = [value for value in confinement_rates if value is not None]
            pd.DataFrame({'change_rates': confinement_rates}).to_excel(writer, sheet_name=f'change_rates_{mean_by_roi}', index=False)

            browian_portions = []
            confined_portions = []
            #Check this part of the code
            brownian_values_per_roi = defaultdict(lambda: [])
            confined_values_per_roi = defaultdict(lambda: [])
            for result in list(Trajectory._get_collection().find(filter_query, {'_id':1,'t':1,'x':1,'y':1,'info.analysis.confinement-states':1})):
                if result.get('info',{},).get('analysis',{}).get('confinement-states',{}) is not None:
                    a_trajectory = Trajectory(
                        x=result['x'],
                        y=result['y'],
                        t=result['t'],
                        info={'analysis' : {'confinement-states': result['info']['analysis']['confinement-states']}},
                        noisy=True
                    )
                    sub_trajectories_by_state = a_trajectory.sub_trajectories_trajectories_from_confinement_states(use_info=True)

                    for sub_trajectory in sub_trajectories_by_state[0]:
                        if not mean_by_roi:
                            browian_portions.append(sub_trajectory.duration)
                        else:
                            browian_portions.append((sub_trajectory.duration, sub_trajectory.info['file'],sub_trajectory.info['roi']))
                    
                    for sub_trajectory in sub_trajectories_by_state[1]:
                        if not mean_by_roi:
                            confined_portions.append(sub_trajectory.duration)
                        else:
                            confined_portions.append((sub_trajectory.duration, sub_trajectory.info['file'],sub_trajectory.info['roi']))

            if mean_by_roi:
                browian_portions = reduce_mean_by_file_and_roi(browian_portions)
                confined_portions = reduce_mean_by_file_and_roi(confined_portions)

            pd.DataFrame({'browian_portions': [p for p in browian_portions if p != 0]}).to_excel(writer, sheet_name=f'browian_portions_{mean_by_roi}', index=False)
            pd.DataFrame({'confined_portions': [p for p in confined_portions if p != 0]}).to_excel(writer, sheet_name=f'confined_portions_{mean_by_roi}', index=False)
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

            for true_unpack_fields in [
                'number_of_trajectories_per_overlap', 'confinement-a', 'confinement-e','confinement-area',
                'non-confinement-steps', 'confinement-steps', 'non-confinement-betha', 'confinement-betha',
                'non-confinement-k', 'confinement-k', 'non-confinement-d_2_4', 'confinement-d_2_4']:
                upload_data_to_writer(writer, filter_query, false_unpack_fields, mean_by_roi, True)

            list_of_confinement_areas_centroids = get_list_of_values_of_analysis_field(filter_query, 'confinement_areas_centroids', mean_by_roi=True)
            list_of_confinement_areas_centroids = list(itertools.chain.from_iterable([pdist(np.array(confinement_areas_centroids) * 1e3).tolist() for confinement_areas_centroids in list_of_confinement_areas_centroids if len(confinement_areas_centroids) >= 2]))
            pd.DataFrame({'confinement_areas_distance': list_of_confinement_areas_centroids}).to_csv(f'./Results/{dataset}_{index}_confinement_areas_distance.csv')
    exit()
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
