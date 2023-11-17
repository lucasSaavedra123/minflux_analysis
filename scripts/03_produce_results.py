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

for dataset in DATASETS_LIST:
    with pd.ExcelWriter(f"./Results/{dataset}_basic_information.xlsx") as writer:
        pd.DataFrame({'k': get_list_of_values_of_analysis_field(dataset, 'k', apply_immobile_criteria=APPLY_GS_CRITERIA)}).to_excel(writer, sheet_name='k', index=False)
        pd.DataFrame({'betha': get_list_of_values_of_analysis_field(dataset, 'betha', apply_immobile_criteria=APPLY_GS_CRITERIA)}).to_excel(writer, sheet_name='betha', index=False)
        pd.DataFrame({'dc': get_list_of_values_of_analysis_field(dataset, 'directional_coefficient', apply_immobile_criteria=APPLY_GS_CRITERIA)}).to_excel(writer, sheet_name='directional_coefficient', index=False)

        residence_times = get_list_of_values_of_analysis_field(dataset, 'residence_time', apply_immobile_criteria=APPLY_GS_CRITERIA)
        residence_times = [residence_time for residence_time in residence_times if residence_time != 0]
        pd.DataFrame({'residence_time': residence_times}).to_excel(writer, sheet_name='residence_time', index=False)

        pd.DataFrame({'ratio': get_list_of_values_of_field(dataset, 'ratio', apply_immobile_criteria=False)}).to_excel(writer, sheet_name='ratio', index=False)

        list_of_trajectories_time = get_list_of_main_field(dataset, 't', apply_immobile_criteria=False)

        intervals = list(itertools.chain.from_iterable([np.diff(time_list) for time_list in list_of_trajectories_time]))
        intervals = [interval for interval in intervals if interval != 0]
        #pd.DataFrame({'interval': intervals}).to_excel(writer, sheet_name='interval', index=False)

        list_of_semi_major_axis = get_list_of_values_of_analysis_field(dataset, 'confinement-a', apply_immobile_criteria=True)
        list_of_semi_major_axis = list(itertools.chain.from_iterable([semi_major_axis for semi_major_axis in list_of_semi_major_axis]))
        pd.DataFrame({'semi_major_axis': list_of_semi_major_axis}).to_excel(writer, sheet_name='semi_major_axis', index=False)

        list_of_eccentricities = get_list_of_values_of_analysis_field(dataset, 'confinement-e', apply_immobile_criteria=True)
        list_of_eccentricities = list(itertools.chain.from_iterable([eccentricities for eccentricities in list_of_eccentricities]))
        pd.DataFrame({'eccentricity': list_of_eccentricities}).to_excel(writer, sheet_name='eccentricity', index=False)

        list_of_confinement_areas = get_list_of_values_of_analysis_field(dataset, 'confinement-area', apply_immobile_criteria=True)
        list_of_confinement_areas = list(itertools.chain.from_iterable([[area * 1e6 for area in areas] for areas in list_of_confinement_areas]))
        pd.DataFrame({'area': list_of_confinement_areas}).to_excel(writer, sheet_name='area', index=False)

        lengths = [len(time_list) for time_list in list_of_trajectories_time]
        lengths = [length for length in lengths if length != 1]
        pd.DataFrame({'length': lengths}).to_excel(writer, sheet_name='length', index=False)

        durations = [time_list[-1] - time_list[0] for time_list in list_of_trajectories_time]
        durations = [duration for duration in durations if duration != 0]
        pd.DataFrame({'duration': durations}).to_excel(writer, sheet_name='duration', index=False)

        basic_info_file.write(f"{dataset} -> n={len(lengths)}\n")
        basic_info_file.write(f"{dataset} -> Time Interval Mean: {np.mean(np.array(intervals)*1e6)}us, S.E.M: {sem(np.array(intervals)*1e6)}us\n")
        basic_info_file.write(f"{dataset} -> Residence Time Scale: {np.mean(residence_times)}us, S.E.M: {sem(residence_times)}s\n")

        list_of_confinement_areas_centroids = get_list_of_values_of_analysis_field(dataset, 'confinement_areas_centroids', apply_immobile_criteria=True)
        list_of_confinement_areas_centroids = list(itertools.chain.from_iterable([pdist(np.array(confinement_areas_centroids) * 1e3).tolist() for confinement_areas_centroids in list_of_confinement_areas_centroids if len(confinement_areas_centroids) >= 2]))
        pd.DataFrame({'confinement_areas_distance': list_of_confinement_areas_centroids}).to_csv(f'./Results/{dataset}_confinement_areas_distance.csv')

    with pd.ExcelWriter(f"./Results/{dataset}_angles_information.xlsx") as writer:
        all_angles = default_angles()
        
        for label in DIFFUSION_BEHAVIOURS_INFORMATION:
            label_angle_information = default_angles()

            ids = get_ids_of_trayectories_under_betha_limits(
                dataset,
                DIFFUSION_BEHAVIOURS_INFORMATION[label]['range_0'],
                DIFFUSION_BEHAVIOURS_INFORMATION[label]['range_1'],
                APPLY_GS_CRITERIA
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

            filter_query = {'info.dataset': dataset, 'info.immobile': False} if APPLY_GS_CRITERIA else {'info.dataset': dataset}
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

    plt.plot(x,pdfs, color=DATASET_TO_COLOR[dataset])
    plt.hist(residence_times, density=True, bins='auto', histtype='stepfilled', alpha=0.2, color=DATASET_TO_COLOR[dataset])

basic_info_file.close()
DatabaseHandler.disconnect()

plt.xlabel('Confinement Time [s]', fontname="Arial", fontsize=40)
plt.yticks(visible=False)
plt.xlim([0,4])
plt.xticks(fontname="Arial", fontsize=40)
plt.ylabel('Frequency', fontname="Arial", fontsize=40)

plt.show()
