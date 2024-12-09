import pandas as pd

import numpy as np
import tqdm
from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from scipy.spatial import ConvexHull

from Trajectory import Trajectory


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

results = {'label':[],'state':[],'area':[]}

p = {
    'x': 1,
    'y': 1,
    't': 1,
    'info.analysis.confinement-classification': 1,
    'info.classified_experimental_condition': 1,
    'info.dataset': 1,
}

for t_info_index, t_info in tqdm.tqdm(enumerate(Trajectory._get_collection().find({'info.immobile':False}, p))):
    fake_trajectory = Trajectory(
        x=t_info['x'],
        y=t_info['y'],
        t=t_info['t'],
        noisy=True
    )

    classifications = t_info['info'].get('analysis', {}).get('confinement-classification', [])

    for index, sub_t in enumerate(fake_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1]):
        #if classifications[index] is not None and f'{classifications[index]}_{t_info_index}_{index}' in ['TD_3_3', 'HD_6_1']:
        #    sub_t.animate_plot(roi_size=0.25, save_animation=True, title=f'{classifications[index]}_{t_info_index}_{index}')
        if classifications[index] is not None:
            raw_trajectory = np.zeros((sub_t.length, 2))
            raw_trajectory[:,0] = sub_t.get_noisy_x() * 1000
            raw_trajectory[:,1] = sub_t.get_noisy_y() * 1000
            area = ConvexHull(raw_trajectory).volume

            results['state'].append(classifications[index])
            results['area'].append(area)

            if 'classified_experimental_condition' in t_info['info']:
                results['label'].append(t_info['info']['dataset']+'_'+t_info['info']['classified_experimental_condition'])
            else:
                results['label'].append(t_info['info']['dataset'])

pd.DataFrame(results).to_csv('r_zones.csv', index=False)

DatabaseHandler.disconnect()