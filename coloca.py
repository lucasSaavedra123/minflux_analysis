from collections import defaultdict
import os
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from spit.colocalize import colocalize_from_locs


PATH = '../Chol and BTX datasets'

def get_trajectory_from_track_dataset(track_dataset, dataset_directory, file_name, track_id):
    info_value = {'dataset': dataset_directory, 'file': file_name, 'trajectory_id': track_id}

    info_value['dcr'] = track_dataset['dcr'].values
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
    dataset = dataset.rename(columns={index: value for index, value in enumerate(['track_id', 't', 'x', 'y', 'intensity', 'dcr'])})

    current_id = dataset.iloc[0]['track_id']
    initial_row = 0
    row_index = 1
    ids_historial = defaultdict(lambda: 0)

    extraction_result = []

    for row_index in tqdm.tqdm(list(range(len(dataset)))):
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

def upload_trajectories_from_file(a_file):
    t = []
    extraction_result = extract_dataframes_from_file(a_file)
    """
    for info_extracted in extraction_result:
        t.append(get_trajectory_from_track_dataset(
            info_extracted[0],
            'test',
            a_file,
            f"{info_extracted[1]}_{info_extracted[2]}"
        ))
    return t
    """
    return [i[0] for i in extraction_result]

files = [
    '231013-105211_mbm test.txt',
    '231013-105628_mbm test-pow8pc.txt',
    '231013-110430_mbm test-pow8pc.txt',
    '231013-111321_mbm test-pow8pc.txt',
    '231013-111726_mbm test-pow8pc.txt',
    '231013-112242_mbm test-pow8pc.txt',
    '231013-112652_mbm test-pow8pc.txt',
    '231013-113251_mbm test-pow8pc.txt',
    '231013-113638_mbm test-pow8pc.txt',
    '231013-124040_mbm test.txt',
    '231013-124511_mbm test.txt',
    '231013-125044_mbm test.txt',
    '231013-125411_mbm test.txt',
    '231013-125818_mbm test.txt',
    '231013-130259_mbm test.txt',
    '231013-130748_mbm test.txt',
    '231013-131100_mbm test.txt',
    '231013-131615_mbm test.txt',
    '231013-131935_mbm test.txt',
    '231013-132310_mbm test.txt',
    '231013-132703_mbm test.txt',
    '231013-153332_mbm test.txt',
    '231013-153631_mbm test.txt',
    '231013-154043_mbm test.txt',
    '231013-154400_mbm test.txt',
    '231013-154702_mbm test.txt',
    '231013-154913_mbm test.txt',
    '231013-155220_mbm test.txt',
    '231013-155616_mbm test.txt',
    '231013-155959_mbm test.txt',
    '231013-160351_mbm test.txt',
    '231013-160951_mbm test.txt',
    '231013-161302_mbm test.txt',
    '231013-161554_mbm test.txt',
    '231013-162155_mbm test.txt',
    '231013-162602_mbm test.txt',
    '231013-162934_mbm test.txt',
    '231013-163124_mbm test.txt',
    '231013-163414_mbm test.txt',
    '231013-163548_mbm test.txt'
]


for file_name in tqdm.tqdm(os.listdir(PATH)):
    dataframes_from_file = upload_trajectories_from_file(os.path.join(PATH,file_name))

    for df_index, _ in enumerate(dataframes_from_file):
        dataframes_from_file[df_index]['channel'] = 1 if dataframes_from_file[df_index]['dcr'].mean() > TDCR_THRESHOLD else 0 #1->Chol, 0->BTX
        dataframes_from_file[df_index]['sx'] = 0
        dataframes_from_file[df_index]['sy'] = 0
        dataframes_from_file[df_index]['bg'] = 0
        dataframes_from_file[df_index]['lpx'] = 0
        dataframes_from_file[df_index]['ellipticity'] = 0
        dataframes_from_file[df_index]['net_gradient'] = 0
        dataframes_from_file[df_index]['loc_precision'] = 0
        dataframes_from_file[df_index]['nearest_neighbor'] = 0
        dataframes_from_file[df_index]['cell_id'] = 0

    final_dataframe = pd.concat(dataframes_from_file)
    #final_dataframe = final_dataframe[final_dataframe['y'] > 1.12e-6]
    #final_dataframe = final_dataframe[final_dataframe['x'] > -5.65e-6]
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(final_dataframe['x'], final_dataframe['t'], final_dataframe['y'], c=final_dataframe['channel'], s=1)
    #plt.scatter(final_dataframe['x'], final_dataframe['y'], c=final_dataframe['channel'])
    #plt.show()
    final_dataframe['t'] -= final_dataframe['t'].min()
    final_dataframe['t'] = final_dataframe['t']//(1e-3)
    result = colocalize_from_locs(final_dataframe[final_dataframe['channel'] == 0].copy(), final_dataframe[final_dataframe['channel'] == 1].copy(), 0.030)
    print(result)
