#It needs tensorflow 2.13 to Run
import numpy as np
import tqdm
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from utils import *

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

def build_dgn(number_of_states):
  input = Input(shape=(None,1))
  x = Bidirectional(LSTM(32, return_sequences=True))(input)
  x = Dense(number_of_states, activation='softmax')(x)
  hid_state = x

  x = Bidirectional(LSTM(16, return_sequences=True))(x)
  output = Dense(1, activation='linear')(x)

  model = Model(input, [hid_state,output])
  model.compile(optimizer=Adam(), loss = ['sparse_categorical_crossentropy','mse'])
  #model = Model(input, output)
  #model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), loss = 'mse')
  return model

def smooth_classifications(classifications):
    result = classifications.copy()
    for i in range(1, len(classifications) - 1):
        if classifications[i - 1] == classifications[i + 1] and classifications[i] != classifications[i - 1]:
            result[i] = classifications[i - 1]
    return result

model = build_dgn(5)
model.load_weights("dgn_for_minflux.keras")

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

for traj in Trajectory.objects():
    if 'intensity' not in traj['info']:
      continue

    t = np.array(traj['t'])
    efo = np.array(traj['info']['intensity'])
    eco = np.diff(t) * efo[:-1]

    if len(eco) > 0:
      eco = np.expand_dims(np.expand_dims(np.array(eco), -1), 0)/275
      y_c, y_o = model.predict(eco)
      y_c = np.argmax(y_c,axis=-1)[0]
      y_o = y_o[0,:,0]

      classifications = smooth_classifications(y_c)

      traj['info']['analysis']['eco_cleaned'] = y_o.tolist()
      traj['info']['analysis']['eco_states'] = classifications.tolist()
      traj['info']['analysis']['eco_state_0'] = (classifications==0).astype(int).tolist()
      traj['info']['analysis']['eco_state_1'] = (classifications==1).astype(int).tolist()
      traj['info']['analysis']['eco_state_2'] = (classifications==2).astype(int).tolist()
      traj['info']['analysis']['eco_state_3'] = (classifications==3).astype(int).tolist()
      traj['info']['analysis']['eco_state_4'] = (classifications==4).astype(int).tolist()
      traj.save()

DatabaseHandler.disconnect()
