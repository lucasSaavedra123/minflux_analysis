import numpy as np
import tqdm
from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
import numpy as np


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

p = {
    '_id': 1,
    'x':1,
    'y':1,
    't': 1,
    'info.analysis.directed-deviation': 1,
    'info.analysis.directed-states-deepspt': 1,
}


t_classifications = []
for t_info in tqdm.tqdm(Trajectory._get_collection().find({}, p)):#find({'info.immobile':True}, p)):
    if 'analysis' not in t_info['info'] or 'directed-states-deepspt' not in t_info['info']['analysis'] or len(t_info['x']) <= 1:
        continue

    fake_trajectory = Trajectory(
        x=t_info['x'],
        y=t_info['y'],
        t=t_info['t'],
        noisy=True
    )

    confinement_classifications = []

    for sub_t in fake_trajectory.sub_trajectories_trajectories_from_confinement_states(custom_states=t_info['info']['analysis']['directed-states-deepspt'])[1]:
        positions = np.zeros((sub_t.length,2))
        positions[:,0] = sub_t.get_noisy_x()
        positions[:,1] = sub_t.get_noisy_y()
        ds = []
        for t_i in range(1,sub_t.length-1):
            d = np.linalg.norm(np.cross(positions[-1]-positions[0], positions[0]-positions[t_i]))/np.linalg.norm(positions[-1]-positions[0])
            ds.append(d)
        
        confinement_classifications.append(np.mean(ds))

    Trajectory._get_collection().update_one({'_id':t_info['_id']}, {"$set":{"info.analysis.directed-deviation":confinement_classifications}})

DatabaseHandler.disconnect()