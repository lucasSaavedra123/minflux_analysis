#THIS SCRIPT UPLOAD INTENSITY TO ALREADY UPLOADED TRAJECTORIES!!!
import os
from collections import defaultdict
import glob

import pandas as pd
import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, 'MINFLUX_DATA')

def get_trajectory_from_track_dataset(track_dataset, file_name, track_id):
    info_value = {'file': file_name, 'trajectory_id': track_id}

    if 'dcr' in track_dataset.columns:
        info_value['dcr'] = track_dataset['dcr'].values
    if 'intensity' in track_dataset.columns:
        info_value['intensity'] = track_dataset['intensity'].values

    return Trajectory(
        x = track_dataset['x'].values * 1e6,
        y = track_dataset['y'].values * 1e6,
        t = track_dataset['t'].values,
        info=info_value,
        noisy=True
    )

def extract_dataframes_from_file(a_file):
    dataset = pd.read_csv(a_file, sep=' ', header=None)

    if len(dataset.columns) == 6:
        dataset = dataset.rename(columns={index: value for index, value in enumerate(['track_id', 't', 'x', 'y', 'intensity', 'dcr'])})
    elif len(dataset.columns) == 5:
        dataset = dataset.rename(columns={index: value for index, value in enumerate(['track_id', 't', 'x', 'y', 'dcr'])})
    elif len(dataset.columns) == 4:
        dataset = dataset.rename(columns={index: value for index, value in enumerate(['track_id', 't', 'x', 'y'])})
    else:
        raise Exception('No valid number of columns')

    current_id = dataset.iloc[0]['track_id']
    initial_row = 0
    row_index = 1
    ids_historial = defaultdict(lambda: 0)

    extraction_result = []

    for row_index in list(range(len(dataset))):
        if dataset.iloc[row_index]['track_id'] != current_id:
            extraction_result.append((
                dataset.iloc[initial_row:row_index].copy().sort_values('t', ascending=True),
                int(current_id),
                int(ids_historial[current_id]),
            ))

            ids_historial[current_id] += 1
            initial_row = row_index
            current_id = dataset.iloc[row_index]['track_id']



    extraction_result.append((
        dataset.iloc[initial_row:row_index].copy().sort_values('t', ascending=True),
        int(current_id),
        int(ids_historial[current_id]),
    ))

    return extraction_result


for a_file in tqdm.tqdm(glob.glob("../AChR data/*.txt")):
    if Trajectory._get_collection().count_documents({'info.file': a_file.split('\\')[-1]}) != 0:
        extraction_result = extract_dataframes_from_file(a_file)

        for info_extracted in extraction_result:
            if not info_extracted[0].empty:
                trajectory = get_trajectory_from_track_dataset(
                    info_extracted[0],
                    a_file.split('\\')[-1],
                    f"{info_extracted[1]}_{info_extracted[2]}"
                )
                
                trajectory_query = Trajectory.objects(info__file=trajectory.info['file'], info__trajectory_id=trajectory.info['trajectory_id'])
                if len(trajectory_query) == 1 and trajectory_query[0].length == trajectory.length:
                    trajectory_query = trajectory_query[0]
                    trajectory_query.info['intensity'] = trajectory.info['intensity']
                    trajectory_query.save()

print(Trajectory._get_collection().count_documents({'info.intensity': {'$exists': True}})/Trajectory._get_collection().count_documents({}))

DatabaseHandler.disconnect()