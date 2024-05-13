import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import IP_ADDRESS, DATASET_TO_DELTA_T

import plotly.express as px
import plotly.graph_objects as go

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, 'MINFLUX_DATA')


trajectories = [trajectory for trajectory in Trajectory.objects(info__file='231013-105211_mbm test.txt') if trajectory.length > 1]
print(len(trajectories))

df_as_dict = {
        'x': [],
        'y': [],
        't': [],
        'dcr': []
    }

line_fig = None

for trajectory_id in np.random.choice(len(trajectories), size=250, replace=False):
    trajectory = trajectories[trajectory_id]
    xs = [np.min(trajectory.get_noisy_x()), np.max(trajectory.get_noisy_x())]
    ys = [np.min(trajectory.get_noisy_y()), np.max(trajectory.get_noisy_y())]

    df_as_dict['x'] += trajectory.get_noisy_x().tolist()
    df_as_dict['y'] += trajectory.get_noisy_y().tolist()
    df_as_dict['t'] += trajectory.get_time().tolist()
    df_as_dict['dcr'] += trajectory.info['dcr']

    current_df = pd.DataFrame({
        'x': trajectory.get_noisy_x().tolist(),
        'y': trajectory.get_noisy_y().tolist(),
    })

    aux_fig = px.line(current_df, x="x", y="y")

    color = 'blue' if np.mean(trajectory.info['dcr']) < 0.55 else 'orange'
    #aux_fig.update_traces(line=dict(color = f'rgba({np.random.randint(0,256)},{np.random.randint(0,256)},{np.random.randint(0,256)},0.85)'))
    aux_fig.update_traces(line=dict(color = color))

    if line_fig is None:
        line_fig = aux_fig
    else:
        line_fig = go.Figure(data=line_fig.data + aux_fig.data)

px.scatter(pd.DataFrame(df_as_dict), x="x", y="y", color="dcr", hover_data=["t", "dcr"]).write_html('BTX and Cholesterol - 231013-105211_mbm test - Pointwise Info.html')
line_fig.write_html('BTX and Cholesterol - 231013-105211_mbm test with lines.html')
