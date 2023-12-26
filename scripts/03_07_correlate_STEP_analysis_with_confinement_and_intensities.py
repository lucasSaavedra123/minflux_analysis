import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import warnings

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

    if index < 5:
        continue

    if index < 4:
        trajectories = Trajectory.objects(info__dataset=dataset, info__immobile=False)
    else:
        trajectories = Trajectory.objects(info__classified_experimental_condition=dataset, info__immobile=False)

    intensities_confined, diffusion_confined = np.array([]), np.array([])
    intensities_non_confined, diffusion_non_confined = np.array([]), np.array([])

    for trajectory in trajectories:
        if trajectory.length == 1:
            continue

        confinement_states = trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33, window_size=3, transition_fix_threshold=9, use_info=True)

        for sub_trajectory in confinement_states[0]:
            if 'intensity' in sub_trajectory.info and len(sub_trajectory.info['intensity']) != 0:
                intensities_non_confined = np.append(intensities_non_confined, np.mean(sub_trajectory.info['intensity']))
                diffusion_non_confined = np.append(diffusion_non_confined, np.mean(sub_trajectory.info['analysis']['step_result']))

        for sub_trajectory in confinement_states[1]:
            if 'intensity' in sub_trajectory.info and len(sub_trajectory.info['intensity']) != 0:
                intensities_confined = np.append(intensities_confined, np.mean(sub_trajectory.info['intensity']))
                diffusion_confined = np.append(diffusion_confined, np.mean(sub_trajectory.info['analysis']['step_result']))

    with pd.ExcelWriter(f"./Results/{dataset}_{index}_gs_True_correlation.xlsx") as writer:
        pd.DataFrame({
            'intensity_mean': intensities_confined,
            'diffusion_mean': diffusion_confined,
        }).to_excel(writer, sheet_name='confined', index=False)

        pd.DataFrame({
            'intensity_mean': intensities_non_confined,
            'diffusion_mean': diffusion_non_confined,
        }).to_excel(writer, sheet_name='non-confined', index=False)

DatabaseHandler.disconnect()
