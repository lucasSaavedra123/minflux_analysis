"""
This script upload all trajectories to the MongoDB database.
"""
import pandas as pd
from scipy.io import loadmat
from CONSTANTS import IP_ADDRESS, COLLECTION_NAME_STORM

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory

mat_data = loadmat('all_tracks_thunder_localizer.mat')

dataset = []

dataset.append({'label': 'BTX',
                'exp_cond': 'CDx'})
dataset.append({'label': 'BTX',
                'exp_cond': 'Control'})
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx-Chol'})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx'})
dataset.append({'label': 'mAb',
                'exp_cond': 'Control'})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx-Chol'})

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME_STORM)

for data in dataset:
    dataset_dataframe = {
        'x':[],
        'y':[],
        'traj_id':[],
        'frame':[],
    }

    print(f"Label: {data['label']}, Experimental Condition: {data['exp_cond']}")
    trajectories = Trajectory.objects(info__label=data['label'], info__experimental_condition=data['exp_cond'])

    for traj in trajectories:
        dataset_dataframe['x'].extend(traj.get_noisy_x())
        dataset_dataframe['y'].extend(traj.get_noisy_y())
        dataset_dataframe['frame'].extend((traj.get_time()/0.010).astype(int))
        dataset_dataframe['traj_id'].extend([str(traj.id)] * traj.length)

    pd.DataFrame(dataset_dataframe).to_csv(f'STORM Datasets/{data["label"]}_{data["exp_cond"]}.csv', index=False)


DatabaseHandler.disconnect()
