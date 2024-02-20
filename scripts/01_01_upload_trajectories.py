"""
This script upload all trajectories to the MongoDB database.
"""
import os
from collections import defaultdict

import pandas as pd
import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, 'MINFLUX_DATA')

trajectories_info = list(Trajectory._get_collection().find({}, {
    'info.dataset':1,
    'info.file':1,
    'info.trajectory_id':1
}))

def check_query_equal_to_document(document, query):
    dataset_equal = document['info']['dataset'] == query['info.dataset']
    file_equal = document['info']['file'] == query['info.file']
    trajectory_id_equal = document['info']['trajectory_id'] == query['info.trajectory_id']

    return dataset_equal and file_equal and trajectory_id_equal

def trajectory_was_already_uploaded(dataset, file, trajectory_id):
    a_query = {
        'info.dataset': dataset,
        'info.file': file,
        'info.trajectory_id': trajectory_id,
    }

    #number_of_trajectories = Trajectory._get_collection().count_documents(a_query)
    number_of_trajectories = len([d for d in trajectories_info if check_query_equal_to_document(d, a_query)])

    assert number_of_trajectories <= 1
    return number_of_trajectories == 1

def upload_trajectory_from_track_dataset(track_dataset, dataset_directory, file_name, track_id):
    if not trajectory_was_already_uploaded(dataset_directory, file_name, track_id) and not track_dataset.empty:
        info_value = {'dataset': dataset_directory, 'file': file_name, 'trajectory_id': track_id}

        if 'dcr' in track_dataset.columns:
            info_value['dcr'] = track_dataset['dcr'].values

        Trajectory(
            x = track_dataset['x'].values * 1e6,
            y = track_dataset['y'].values * 1e6,
            t = track_dataset['t'].values,
            info=info_value,
            noisy=True
        ).save()

def extract_dataframes_from_file(dataset_directory, a_file):
    #print(f"Extracting info from dataset {dataset_directory} and file {a_file}")
    
    try:
        dataset = pd.read_csv(os.path.join(dataset_directory,a_file), sep=' ', header=None)
    except pd.errors.EmptyDataError:
        return []

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

def upload_trajectories_from_directory(dataset_directory):
    #print('Uploading Trajectories from dataset directory', dataset_directory)

    for a_file in tqdm.tqdm([file for file in os.listdir(dataset_directory) if file.endswith('.txt')]):
        extraction_result = extract_dataframes_from_file(dataset_directory, a_file)
        #print(f"Uploading trajectories from dataset {dataset_directory} and file {a_file}")
        for info_extracted in extraction_result:
            upload_trajectory_from_track_dataset(
                info_extracted[0],
                dataset_directory,
                a_file,
                f"{info_extracted[1]}_{info_extracted[2]}"
            )

    print('All trajectories uploaded from dataset directory', dataset_directory)

for dataset in DATASETS_LIST:
    print(dataset)
    upload_trajectories_from_directory(dataset)

DatabaseHandler.disconnect()
