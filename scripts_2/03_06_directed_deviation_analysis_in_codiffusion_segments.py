
import pandas as pd
import tqdm
from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
import numpy as np
import pandas as pd
import scipy.stats as stats


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

p = {
    '_id': 1,
    'x':1,
    'y':1,
    't': 1,
    'info.roi': 1,
    'info.file': 1,
    'info.analysis.directed-deviation': 1,
    'info.analysis.directed-states-deepspt': 1,
    'info.analysis.predictions_on_each_directed_segments': 1
}


t_classifications = []

queries = {
    'CF®680R-BTX(+fPEG-Chol)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'BTX680R'},
    'fPEG-Chol(+CF®680R-BTX)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'fPEG-Chol'},
}

directed_deviations_info = []

for category,query in queries.items():
    query['info.immobile'] = False
    for t_info in tqdm.tqdm(Trajectory._get_collection().find(query, p)):#find({'info.immobile':True}, p)):
        if 'analysis' not in t_info['info'] or f'predictions_on_each_directed_segments' not in t_info['info']['analysis']:
            continue

        fake_trajectory = Trajectory(
            x=t_info['x'],
            y=t_info['y'],
            t=t_info['t'],
            noisy=True
        )

        #Iso es que esta isolated, no esta con Chol
        for sub_t, prediction in zip(
            fake_trajectory.sub_trajectories_trajectories_from_confinement_states(custom_states=t_info['info']['analysis']['directed-states-deepspt'])[1],
            t_info['info']['analysis'][f'predictions_on_each_directed_segments']
        ):
            positions = np.zeros((sub_t.length,2))
            positions[:,0] = sub_t.get_noisy_x()
            positions[:,1] = sub_t.get_noisy_y()
            ds = []
            for t_i in range(1,sub_t.length-1):
                d = np.linalg.norm(np.cross(positions[-1]-positions[0], positions[0]-positions[t_i]))/np.linalg.norm(positions[-1]-positions[0])
                ds.append(d)
            
            if prediction['classification'] == 'not_iso':
                directed_deviations_info.append({
                    'dd':np.mean(ds),
                    'roi':t_info['info']['roi'],
                    'file':t_info['info']['file'],
                    'category': category
                })


df = pd.DataFrame(directed_deviations_info)
df = df.groupby(['category','roi','file'],as_index=False).mean()


df_grouped = df.groupby(['category','roi','file'], as_index=False).mean()
df1 = df_grouped[df_grouped['category'] == 'CF®680R-BTX(+fPEG-Chol)']
df2 = df_grouped[df_grouped['category'] == 'fPEG-Chol(+CF®680R-BTX)']

ks_stat, ks_p = stats.ks_2samp(df1['dd'], df2['dd'])
print("p-value:", ks_p)

summary = df_grouped.groupby('category')['dd'].agg(
    mean='mean',
    sem=lambda x: stats.sem(x, nan_policy='omit')
).reset_index()

print("\nSummary:")
print(summary)

DatabaseHandler.disconnect()