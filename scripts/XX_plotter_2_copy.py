import random
from collections import Counter
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import IP_ADDRESS, DATASET_TO_DELTA_T

import plotly.express as px
import plotly.graph_objects as go

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, 'MINFLUX_DATA')


trajectories = [trajectory for trajectory in Trajectory.objects(info__dataset='Cholesterol and btx') if trajectory.length > 1]
print(len(trajectories))

labels = []

for trajectory in tqdm.tqdm(trajectories):
    labels.append('BTX680R' if np.mean(trajectory.info['dcr']) < 0.55 else 'fPEG-Chol')

print(Counter(labels))