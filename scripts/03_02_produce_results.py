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

new_datasets_list = DATASETS_LIST.copy()
new_datasets_list = DATASETS_LIST[:-1]
new_datasets_list.append(BTX_NOMENCLATURE)
new_datasets_list.append(CHOL_NOMENCLATURE)

for index, dataset in enumerate(new_datasets_list):
    print(dataset)
    SEARCH_FIELD = 'info.dataset' if index < 4 else 'info.classified_experimental_condition'
    with pd.ExcelWriter(f"./Results/{dataset}_{index}_gs_{APPLY_GS_CRITERIA}_basic_information.xlsx") as writer:
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

        lengths = [len(time_list) for time_list in list_of_trajectories_time]
        lengths = [length for length in lengths if length != 1]
        pd.DataFrame({'length': lengths}).to_excel(writer, sheet_name='length', index=False)

        durations = [time_list[-1] - time_list[0] for time_list in list_of_trajectories_time]
        durations = [duration for duration in durations if duration != 0]
        pd.DataFrame({'duration': durations}).to_excel(writer, sheet_name='duration', index=False)

        #Data that takes all mobile trajectories
        filter_query = {SEARCH_FIELD: dataset, 'info.immobile': False}

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

        list_of_confinement_areas_centroids = get_list_of_values_of_analysis_field(filter_query, 'confinement_areas_centroids')
        list_of_confinement_areas_centroids = list(itertools.chain.from_iterable([pdist(np.array(confinement_areas_centroids) * 1e3).tolist() for confinement_areas_centroids in list_of_confinement_areas_centroids if len(confinement_areas_centroids) >= 2]))
        pd.DataFrame({'confinement_areas_distance': list_of_confinement_areas_centroids}).to_csv(f'./Results/{dataset}_{index}_confinement_areas_distance.csv')

basic_info_file = open('./Results/basic_info.txt','w')
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
