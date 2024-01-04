"""
Upong STEP analysis, intensity and diffusion coefficient
of all experiment within and out confinement zones
are produced
"""
import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import warnings

import tqdm

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *

warnings.filterwarnings('error') 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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

    trajectory_info = list(Trajectory._get_collection().find(filter_query, {f'info.analysis.step_result.non-confinement': 1, '_id': 0}))
    diffusion_non_confined = list(itertools.chain.from_iterable([i['info']['analysis']['step_result']['non-confinement'] for i in trajectory_info]))
    trajectory_info = list(Trajectory._get_collection().find(filter_query, {f'info.analysis.step_result.confinement': 1, '_id': 0}))
    diffusion_confined = list(itertools.chain.from_iterable([i['info']['analysis']['step_result']['confinement'] for i in trajectory_info]))

    with pd.ExcelWriter(f"./Results/{dataset}_{index}_gs_True_correlation.xlsx") as writer:
        pd.DataFrame({
            'diffusion_mean': diffusion_confined,
        }).to_excel(writer, sheet_name='confined', index=False)

        pd.DataFrame({
            'diffusion_mean': diffusion_non_confined,
        }).to_excel(writer, sheet_name='non-confined', index=False)

DatabaseHandler.disconnect()
