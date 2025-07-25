"""
This script upload all trajectories to the MongoDB database.
"""
import glob

from matplotlib import pyplot as plt
import pandas as pd
from CONSTANTS import IP_ADDRESS, COLLECTION_NAME_STORM

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory


for predicted_file in glob.glob('./STORM Datasets/*full_prediction_qsr.csv'):
    label, exp_cond = predicted_file.split('.csv')[0].split('_')
    print(f"Label: {label}, Experimental Condition: {exp_cond}")

    dataset = pd.read_csv(predicted_file)
    for cluster_id in dataset['cluster_id_predicted'].unique():
        cluster = dataset[dataset['cluster_id_predicted'] == cluster_id]

        if len(cluster['traj_id'].unique()) == 1:
            traj_id = cluster['traj_id'].unique()[0]
            dataset.loc[dataset['traj_id'] == traj_id,'cluster_id_predicted'] = 0
            dataset.loc[dataset['traj_id'] == traj_id,'clusterized_predicted'] = 0

    #plt.scatter(dataset['x'], dataset['y'], c=dataset['clusterized_predicted'])
    #plt.show()

    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME_STORM)
    for traj_id in dataset['traj_id'].unique():
        traj = dataset[dataset['traj_id'] == traj_id]
        traj = traj.sort_values('frame')
        trajectory = Trajectory.objects(id=traj_id)[0]
        trajectory.info['analysis']['clustered_state'] = traj['clusterized_predicted'].tolist()
        trajectory.save()
    DatabaseHandler.disconnect()
