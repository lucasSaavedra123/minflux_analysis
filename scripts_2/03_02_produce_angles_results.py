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


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

bin_width = 20

filter_query = {'info.immobile': False}

with pd.ExcelWriter(f"./Results/deepspt_turning_angle_gs_True_angles_information.xlsx") as writer:
    all_angles = default_angles()
    
    for state in ['directed','normal','subdifussive','confinement']:
        label_angle_information = default_angles()

        angles_infos = Trajectory._get_collection().find(
            filter_query,
            {f'info.analysis.angles_by_state_deepspt.{state}':1}
        )

        for angle_info in angles_infos:
            if 'analysis' in angle_info['info']:
                for angle in angle_info['info']['analysis']['angles_by_state_deepspt'][state]['angles']:
                    label_angle_information[angle] += angle_info['info']['analysis']['angles_by_state_deepspt'][state]['angles'][angle]
                    all_angles[angle] += angle_info['info']['analysis']['angles_by_state_deepspt'][state]['angles'][angle]

        for angle in label_angle_information:
            frequency, bin_edges = custom_histogram(np.array(label_angle_information[angle]), 0, 180, bin_width)
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
        new_angle_information.to_excel(writer, sheet_name=state, index=False)

    for angle in all_angles:
        frequency, bin_edges = custom_histogram(np.array(all_angles[angle]), 0, 180, bin_width)
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

DatabaseHandler.disconnect()
