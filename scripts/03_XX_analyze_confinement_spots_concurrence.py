import pandas as pd
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np

from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory


FILE_AND_ROI_FILE_CACHE = 'file_and_roi.txt'
CENTROIDS_FILE_CACHE = 'centroids.npy'

if not os.path.exists(CENTROIDS_FILE_CACHE):
    def unique(list1):
    
        # initialize a null list
        unique_list = []
    
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)

        return unique_list

    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    """
    file_id_and_roi_list = [[r['info']['file'], r['info']['roi']] for r in Trajectory._get_collection().find(
        {},
        {f'id':1, 'info.roi':1, 'info.file':1}
    )]
    """

    if not os.path.exists(FILE_AND_ROI_FILE_CACHE):
        file_id_and_roi_list = [[r['info']['file'], r['info']['roi']] for r in Trajectory._get_collection().find(
            {},
            {f'id':1, 'info.roi':1, 'info.file':1}
        )]

        file_id_and_roi_list = unique(file_id_and_roi_list)

        a_file = open(FILE_AND_ROI_FILE_CACHE, 'w')
        for file, roi in file_id_and_roi_list:
            a_file.write(f'{file},{roi}\n')
        a_file.close()
    else:
        file_id_and_roi_list = []
        a_file = open(FILE_AND_ROI_FILE_CACHE, 'r')
        for line in a_file.readlines():
            line = line.strip()
            line = line.split(',')
            file_id_and_roi_list.append([line[0], int(line[1])])
        a_file.close()

    p = {
        'x':1,
        'y':1,
        'info.trajectory_id': 1,
        'info.analysis.confinement-states': 1
    }

    areas = []
    confinement_centroids = []

    for i, file_id_and_roi in tqdm.tqdm(list(enumerate(file_id_and_roi_list))):
        dataframe = {
            'x': [],
            'y': [],
            'track_id': [],
            'confined': []
        }

        for raw_trajectory in Trajectory._get_collection().find({'info.roi':file_id_and_roi[1], 'info.file':file_id_and_roi[0]}, p):
            trajectory_length = len(raw_trajectory['x'])
            dataframe['x'] += raw_trajectory['x']
            dataframe['y'] += raw_trajectory['y']
            dataframe['track_id'] += [raw_trajectory['info']['trajectory_id']] * trajectory_length
            dataframe['confined'] += raw_trajectory['info'].get('analysis', {}).get('confinement-states', [0] * trajectory_length)

        dataframe = pd.DataFrame(dataframe)

        width = dataframe['x'].max() - dataframe['x'].min()
        height = dataframe['y'].max() - dataframe['y'].min()

        dataframe['x'] = dataframe['x'] - dataframe['x'].min()
        dataframe['y'] = dataframe['y'] - dataframe['y'].min()

        dataframe['x'] -= width/2
        dataframe['y'] -= height/2

        current_area = width*height

        if current_area < 30:

            for trajectory_id in dataframe['track_id'].unique():
                trajectory_dataframe = dataframe[dataframe['track_id'] == trajectory_id]
                trajectory_dataframe = trajectory_dataframe[trajectory_dataframe['confined'] == 1]

                if len(trajectory_dataframe) != 0 and -2.5 < trajectory_dataframe['x'].mean() < 2.5 and -2.5 < trajectory_dataframe['y'].mean() < 2.5:
                    confinement_centroids += [[trajectory_dataframe['x'].mean(), trajectory_dataframe['y'].mean()]]

                #plt.scatter([a[0] for a in confinement_centroids], [a[1] for a in confinement_centroids], color='red', marker='X')
            
    #plt.show()

    x_confinement_centroids = [a[0] for a in confinement_centroids]
    y_confinement_centroids = [a[1] for a in confinement_centroids]
    centroids = np.transpose(np.array([x_confinement_centroids, y_confinement_centroids]))
    
    np.save(CENTROIDS_FILE_CACHE, centroids)

    DatabaseHandler.disconnect()
else:
    centroids = np.load(CENTROIDS_FILE_CACHE)

import seaborn as sns
sns.set_style("white")
sns.kdeplot(x=centroids[:,0], y=centroids[:,1], cmap="Reds", fill=True, levels=10, bw_adjust=1, bw_method=0.05)
plt.show()

#plt.hist2d(x_confinement_centroids, y_confinement_centroids, bins=(30, 30), cmap=plt.cm.jet)
#plt.show()

DatabaseHandler.disconnect()