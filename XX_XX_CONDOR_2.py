import keras
from keras import layers
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import json
import pandas as pd
import numpy as np
import tqdm
from utils import transform_traj_into_features
from CONSTANTS import *
from DatabaseHandler import DatabaseHandler
from sklearn.model_selection import train_test_split
import os
from Trajectory import Trajectory
from collections import defaultdict
import pickle
import glob
from scipy.stats import mode
from collections import Counter


from matplotlib import pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor


def anomalous_diff_mixture(track_len=200,
                        nb_tracks = 100,
                        LocErr=0.007, # localization error in x, y and z (even if not used)
                        nb_states = 1,
                        Fs = np.array([1]),
                        Ds = np.array([0.05]),
                        nb_dims = 2,
                        velocities = np.array([0.1]),
                        angular_Ds = np.array([0.0]),
                        conf_forces = np.array([0.2]),
                        conf_Ds = np.array([0.10]),
                        conf_dists = np.array([0.0]),
                        LocErr_std = 0,
                        dt = 0.001,
                        nb_sub_steps = 10,
                        nb_blurring_sub_steps = 1):
    
    #nb_blurring_sub_steps = min(max(int(np.ceil(0*nb_sub_steps)), 1), nb_sub_steps)
    
    # diff + persistent motion + elastic confinement
    conf_sub_forces = conf_forces / nb_sub_steps
    sub_dt = dt / nb_sub_steps
    sub_step_wise_loc_err = LocErr*nb_blurring_sub_steps**0.5

    cum_Fs = np.zeros(nb_states)
    cum_Fs[0] = Fs[0]
    for state in range(1, nb_states):
        cum_Fs[state] = cum_Fs[state-1] + Fs[state]
    
    for kkk in range(nb_tracks):
        state = np.argmin(np.random.rand()>cum_Fs)
        
        D, velocity, angular_D, conf_sub_force, conf_D, conf_dist = (Ds[state], velocities[state], angular_Ds[state], conf_sub_forces[state], conf_Ds[state], conf_dists[state])
    
        positions = np.zeros((track_len * nb_sub_steps, nb_dims))
        disps = np.random.normal(0, np.sqrt(2*D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
    
        anchor_positions = np.random.normal(0, np.sqrt(2*conf_D*sub_dt), ((track_len) * nb_sub_steps - 1, nb_dims))
        anchor_positions[0] = positions[0] + np.random.normal(0,conf_dist, nb_dims)
    
        for i in range(1, len(anchor_positions)):
            anchor_positions[i] += anchor_positions[i-1]
    
        d_angles = np.random.normal(0, 1, ((track_len) * nb_sub_steps)-1) * (2*angular_D*sub_dt)**0.5
        angles = np.zeros((track_len * nb_sub_steps-1))
        angles[0] = np.random.rand()*2*np.pi
        for i in range(1, len(d_angles)):
            angles[i] = angles[i-1] + d_angles[i]

        for i in range(len(positions)-1):
            angle = angles[i-1]
            pesistent_disp = np.array([np.cos(angle), np.sin(angle)]).T * velocity/nb_sub_steps
            positions[i+1] = positions[i] + pesistent_disp + disps[i]
            positions[i+1] = (1-conf_sub_force) *  positions[i+1] + conf_sub_force * anchor_positions[i]
        
        positions += np.random.normal(0, sub_step_wise_loc_err, (track_len*nb_sub_steps, nb_dims))
        final_track = np.zeros((track_len, nb_dims))
        for i in range(track_len):
            #final_track[i] = np.mean(positions[i*nb_sub_steps:i*nb_sub_steps+nb_blurring_sub_steps], axis=0)
            #final_track[i] = positions[i*nb_sub_steps]
            for j in range(len(final_track[i])):
                final_track[i, j] = np.mean(positions[i*nb_sub_steps:i*nb_sub_steps+nb_blurring_sub_steps, j])
        
        #if nb_blurring_sub_steps==1:
        #    final_track += np.random.normal(0, LocErr, (track_len, nb_dims))
    
        if kkk == 0:
            final_tracks = [final_track]
            states = [state]
        else:
            final_tracks.append(final_track)
            states.append(state)
    return final_tracks, states

Y = {'c_f':[],'d_p':[]}
X = []

for _ in tqdm.tqdm(list(range(10_000))):
    length = int(np.random.uniform(25,1000))
    new_array = np.zeros((1,length,3))
    dt = np.random.uniform(0.010,0.0001)
    d = np.random.uniform(0.01,2)
    confinement_d = 0#np.random.uniform(0.001,1)
    confinement_factor = np.random.uniform(0,1)

    tracks, _ = anomalous_diff_mixture(
        track_len=length, # Number of time points per track
        nb_tracks = 1, # Number of tracks
        LocErr=0.007, # localization error in x, y and z (even if not used)
        nb_states = 1, # number of states
        Fs = np.array([1.0]), # Fractions of each state
        Ds = np.array([d]), # Diffusion coefficients of each state
        nb_dims = 2,
        velocities = np.array([0.0]), # Velocities coefficients of each state
        angular_Ds = np.array([0.0]), # Rotational diffusion coefficient of the velocity for each state
        conf_forces = np.array([confinement_factor]), # Confinement factor for each state
        conf_Ds = np.array([confinement_d]), # Diffusion coefficients of the potential well for each state
        conf_dists = np.array([0.0]), # standard deviation of the initial distance between the first position and the first potential well center
        dt = dt, # time frame
        nb_sub_steps = 1, # number of sub-steps per step
        nb_blurring_sub_steps = 1) # Number of substeps used to account for dynamic localization errors

    new_array[0,:,0] = tracks[0][:,0]
    new_array[0,:,1] = tracks[0][:,1]
    new_array[0,:,2] = np.arange(length)*dt

    X.append(transform_traj_into_features(new_array)[0])
    Y['c_f'].append(confinement_factor)

X = np.array(X)
Y = np.array(Y['c_f'])

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20)

regr = DecisionTreeRegressor()
regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_val)

plt.rcParams.update({'font.size': 18})
plt.hist2d(Y_val,Y_pred,bins=25,range=[[0,1],[0,1]], cmap='PuBu')
plt.xlabel('Ground-Truth confinement factor')
plt.ylabel('Predicted confinement factor')
plt.subplots_adjust(left=0.36, right=0.92, bottom=0.16, top=0.90)

plt.savefig('confinement_factor_performance.svg')

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

p = {
    '_id': 1,
    'x':1,
    'y':1,
    't': 1,
    'info.analysis.confinement-factor': 1,
    'info.analysis.confinement-states-deepspt': 1,
}

t_classifications = []
for t_info in tqdm.tqdm(Trajectory._get_collection().find({}, p)):#find({'info.immobile':True}, p)):
    if 'analysis' not in t_info['info'] or 'confinement-states-deepspt' not in t_info['info']['analysis'] or len(t_info['x']) <= 1:
        continue

    fake_trajectory = Trajectory(
        x=t_info['x'],
        y=t_info['y'],
        t=t_info['t'],
        noisy=True
    )

    confinement_classifications = []

    for sub_t in fake_trajectory.sub_trajectories_trajectories_from_confinement_states(custom_states=t_info['info']['analysis']['confinement-states-deepspt'])[1]:
        raw_input = np.zeros((1, (60*2)))

        raw_sub_t = np.zeros((1, sub_t.length, 3))
        raw_sub_t[0,:,0] = np.array(sub_t.get_noisy_x())
        raw_sub_t[0,:,1] = np.array(sub_t.get_noisy_y())
        raw_sub_t[0,:,2] = np.array(sub_t.get_time())

        try:
            raw_input[0, :] = transform_traj_into_features(raw_sub_t)[0]
        except:
            confinement_classifications.append(None)
            continue

        confinement_classifications.append(regr.predict(raw_input)[0])

    Trajectory._get_collection().update_one({'_id':t_info['_id']}, {"$set":{"info.analysis.confinement-factor":confinement_classifications}})

DatabaseHandler.disconnect()