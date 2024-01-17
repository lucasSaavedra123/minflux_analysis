from collections import defaultdict
import os
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point


from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from spit.colocalize import colocalize_from_locs
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


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
    #return [i[0] for i in extraction_result]
    for info_extracted in extraction_result:
        t.append(get_trajectory_from_track_dataset(
            info_extracted[0],
            'test',
            a_file,
            f"{info_extracted[1]}_{info_extracted[2]}"
        ))
    return t


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
    trajectories_divided = {'btx':[],'chol':[]}
    chol_confined_zones = []

    #Discriminate trajectories and keep confined zones of Chol
    for t in tqdm.tqdm(upload_trajectories_from_file(os.path.join(PATH,file_name))):
        label = 'chol' if np.mean(t.info['dcr']) > TDCR_THRESHOLD else 'btx'
        if not t.is_immobile(4.295):
            trajectories_divided[label].append(t)

            if label == 'chol':
                for subt_chol in t.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1]:
                    #t.info['analysis'] = {}
                    #t.info['analysis']['confinement-states'] = t.confinement_states(return_intervals=False, v_th=33)
                    chol_confined_zones.append((subt_chol, MultiPoint([p for p in zip(subt_chol.get_noisy_x(), subt_chol.get_noisy_y())]).convex_hull))

    for btx in trajectories_divided['btx']:
        index = 0
        while index < btx.length:            
            for j in chol_confined_zones:
                if j[1].contains(Point(btx.get_noisy_x()[index], btx.get_noisy_y()[index])):
                    t_0 = btx.get_time()[index]
                    index += 1

                    while index < btx.length and j[1].contains(Point(btx.get_noisy_x()[index], btx.get_noisy_y()[index])):
                        index += 1

                    t_1 = btx.get_time()[index-1]
        
                    t = t_1 - t_0

                    plt.plot(np.linspace(t_0, t_1, 100), [1]*100)
                    plt.plot(np.linspace(j[0].get_time()[0], j[0].get_time()[-1], 100), [2]*100)
                    plt.show()
