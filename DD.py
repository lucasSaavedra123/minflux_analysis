import itertools

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import expon, sem
from scipy.spatial.distance import pdist
from bson.objectid import ObjectId
from scipy.stats import sem, kstest

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

new_datasets_list = DATASETS_LIST.copy()
new_datasets_list = DATASETS_LIST[:-1]
new_datasets_list.append(BTX_NOMENCLATURE)
new_datasets_list.append(CHOL_NOMENCLATURE)

for index, dataset in enumerate(new_datasets_list):
    print(dataset)
    SEARCH_FIELD = 'info.dataset' if index < 4 else 'info.classified_experimental_condition'

    if index < 4:
        #trajectories = Trajectory.objects(info__dataset=dataset, info__immobile=False)
        filter_query = {'info.dataset': dataset, 'info.immobile': False}
    else:
        #trajectories = Trajectory.objects(info__classified_experimental_condition=dataset, info__immobile=False)
        filter_query = {'info.classified_experimental_condition': dataset, 'info.immobile': False}

    trajectory_ids = [str(document['_id']) for document in Trajectory._get_collection().find(filter_query, {f'_id':1})]
    times = []

    for trajectory_id in tqdm.tqdm(trajectory_ids):
        t = Trajectory.objects(id=trajectory_id)[0]

        if 'analysis' in t.info and 'residence_time' in t.info['analysis']:
            times.append(t.duration - t.info['analysis']['residence_time'])

    np.savetxt(f"{index}_{dataset}.txt", times)

DatabaseHandler.disconnect()