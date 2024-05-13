import pandas as pd
import tqdm
from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from collections import defaultdict


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

p = {
    '_id': 1,
    'x':1,
    'y':1,
    't': 1,
    'info.dataset': 1,
    'info.classified_experimental_condition': 1,
    'info.analysis.betha': 1,
    'info.analysis.confinement-classification': 1,
}

tracks_by_dataset = defaultdict(lambda: {'x': [],'y': [],'frame': [],'id': []})
sub_id_by_dataset = defaultdict(lambda: 0)


for t_info in tqdm.tqdm(Trajectory._get_collection().find({'info.immobile':False}, p)):
    if 'analysis' not in t_info['info'] or len(t_info['x']) <= 1:
        continue

    if 'classified_experimental_condition' in t_info['info']:
        dataset_label = t_info['info']['dataset']+'_'+t_info['info']['classified_experimental_condition']
    else:
        dataset_label = t_info['info']['dataset']

    fake_trajectory = Trajectory(
        x=t_info['x'],
        y=t_info['y'],
        t=t_info['t'],
        noisy=True
    )

    for sub_t in fake_trajectory.sub_trajectories_trajectories_from_confinement_states(v_th=33)[1]:
        tracks_by_dataset[dataset_label]['x'] += sub_t.get_noisy_x().tolist()
        tracks_by_dataset[dataset_label]['y'] += sub_t.get_noisy_y().tolist()
        tracks_by_dataset[dataset_label]['frame'] += list(range(sub_t.length))
        tracks_by_dataset[dataset_label]['id'] += [sub_id_by_dataset[dataset_label]] * sub_t.length
        sub_id_by_dataset[dataset_label] += 1

for label in tracks_by_dataset:
    pd.DataFrame(tracks_by_dataset[label]).to_csv(f'{label}_confined_tracks.csv', index=False)

DatabaseHandler.disconnect()