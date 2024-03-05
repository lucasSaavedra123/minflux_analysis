"""
This script upload all trajectories to the MongoDB database.
"""
import os
from collections import defaultdict
from matplotlib import pyplot as plt

import pandas as pd
import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, 'MINFLUX_DATA')

trajectories_infos = list(Trajectory._get_collection().find({}, {
    'info.dataset':1,
    'info.file':1,
}))

files_cache = []

for info in trajectories_infos:

    info_alias = os.path.join(info['info']['dataset'], info['info']['file'])

    if info_alias in files_cache:
        continue
    else:
        files_cache.append(info_alias)

    q = {
        'info.dataset': info['info']['dataset'],
        'info.file': info['info']['file'],
    }

    p = {
        'x':1,
        'y':1,
        'info.trajectory_id':1
    }
    
    trajectories_infos = list(Trajectory._get_collection().find(q, p))

    raw_dataframe = {
        'x': [],
        'y': [],
        'trajectory_id': []
    }

    for sub_info in trajectories_infos:
        raw_dataframe['x'] += sub_info['x']
        raw_dataframe['y'] += sub_info['y']
        raw_dataframe['trajectory_id'] += [sub_info['info']['trajectory_id']] * len(sub_info['x'])

    dataframe = pd.DataFrame(raw_dataframe)

    plt.scatter(dataframe['x'], dataframe['y'], s=1)
    plt.show()

DatabaseHandler.disconnect()
