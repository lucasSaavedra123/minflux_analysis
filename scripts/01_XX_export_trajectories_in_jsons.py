"""
All important results like areas, axis lengths, etc. 
are produced within this file.
"""
import numpy as np
import matplotlib.pyplot as plt
import json

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *
import tqdm


APPLY_GS_CRITERIA = True

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

"""
INDIVIDUAL_DATASETS = [
    'Control',
    'CDx',
    'BTX680R',
    'CholesterolPEGKK114',
    'CK666-BTX680',
    'CK666-CHOL',
    'BTX640-CHOL-50-nM',
    'BTX640-CHOL-50-nM-LOW-DENSITY',
]"""

"""[
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]
"""

labels_list = [
    ['BTX680R', 'CholesterolPEGKK114'],
    ['CK666-BTX680', 'CK666-CHOL'],
    ['Cholesterol and btx'],
    ['CK666-BTX680-CHOL']
]

for labels in tqdm.tqdm(labels_list):
    data = {
        'x':[],
        'y':[],
        't':[],
        'id': [],
        'label':[]
    }

    if len(labels) == 2:
        for label in labels:
            raw_trajectories = Trajectory._get_collection().find({'info.dataset': label, 'info.immobile': False}, {f'id':1, 'x':1, 'y':1, 't':1, 'info.file':1, 'info.trajectory_id':1})

            for raw_trajectory in raw_trajectories:
                data['x'] += raw_trajectory['x']
                data['y'] += raw_trajectory['y']
                data['t'] += raw_trajectory['t']
                data['id'] += [raw_trajectory['info']['file']+'_'+raw_trajectory['info']['trajectory_id']] * len(raw_trajectory['x'])
                data['label'] += [label] * len(raw_trajectory['x'])
    elif len(labels) == 1:
        dataset = labels[0]
        labels = [dataset+"-"+CHOL_NOMENCLATURE, dataset+"-"+BTX_NOMENCLATURE]
        for i, label in enumerate(labels):
            raw_trajectories = Trajectory._get_collection().find({'info.dataset': dataset, 'info.immobile': False, 'info.classified_experimental_condition': [CHOL_NOMENCLATURE, BTX_NOMENCLATURE][i]}, {f'id':1, 'x':1, 'y':1, 't':1, 'info.file':1, 'info.trajectory_id':1})

            for raw_trajectory in raw_trajectories:
                data['x'] += raw_trajectory['x']
                data['y'] += raw_trajectory['y']
                data['t'] += raw_trajectory['t']
                data['id'] += [raw_trajectory['info']['file']+'_'+raw_trajectory['info']['trajectory_id']] * len(raw_trajectory['x'])
                data['label'] += [label] * len(raw_trajectory['x'])

    pd.DataFrame(data).to_csv(f"D:\GitHub Repositories\Diffusional-Fingerprinting\{labels[0]}_{labels[1]}.csv")



"""
data = {
    'particle#':[],
    'xPos':[],
    'yPos':[],
}

raw_trajectories = list(Trajectory._get_collection().find({'info.file': '231013-131100_mbm test.txt', 'info.roi': 2, 'info.immobile': False}, {f'id':1, 'x':1, 'y':1, 't':1, 'info.file':1, 'info.trajectory_id':1}))

ids = list(set([r['info']['file']+'_'+r['info']['trajectory_id'] for r in raw_trajectories]))

intervals = []

for raw_trajectory in raw_trajectories:
    data['xPos'] += (np.array(raw_trajectory['x']) * 1000).tolist()
    data['yPos'] += (np.array(raw_trajectory['y']) * 1000).tolist()
    data['particle#'] += [ids.index(raw_trajectory['info']['file']+'_'+raw_trajectory['info']['trajectory_id'])] * len(raw_trajectory['x'])

    intervals += np.diff(raw_trajectory['t']).tolist()

with open('D:\GitHub Repositories\GPDiffusionMapping\data\example.txt', 'w') as f:  
	f.write(str(np.mean(intervals)))

dataframe = pd.DataFrame(data)
dataframe.to_csv("D:\GitHub Repositories\GPDiffusionMapping\data\example.csv", index=False)
"""
DatabaseHandler.disconnect()
