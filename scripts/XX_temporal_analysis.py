import ray
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
import matplotlib as mpl
import ruptures as rpt
from statsmodels.tsa.stattools import acf

import plotly.express as px
import plotly.graph_objects as go

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

try:
    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({'info.immobile':False}, {'_id':1})]

    while True:
        trajectory = Trajectory.objects(id=np.random.choice(uploaded_trajectories_ids))[0]
        
        segments_mean, break_points = trajectory.directional_correlation_segmentation(steps_lag=1,window_size=11, min_size=5, return_break_points=True)
        
        print(segments_mean)
        print(break_points)
        directional_correlation = trajectory.directional_correlation(steps_lag=1,window_size=11)

        plt.plot(directional_correlation, c='grey', linewidth=1)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        #rpt.display(directional_correlation, segments, segments)     
        segment_i = 0
        for segment in segments:
            a_mean = round(np.mean(directional_correlation[segment_i:segment]), 2)
            a_text = r"$\bar{DC_i}="
            a_text += str(a_mean)
            a_text += r'$'

            segment_positions = list(range(segment_i, segment))

            plt.plot(segment_positions, [a_mean] * len(segment_positions))
            plt.text(segment_i, a_mean+0.05, a_text, fontsize=15)
            segment_i = segment

        plt.ylim([-1,1])
        plt.ylim(0, )
        plt.xlabel(r"$i$", fontsize=22)
        plt.ylabel(r"$DC_{i}$", fontsize=22)

        plt.show()

        """
        df = pd.DataFrame({
            'x': trajectory.get_noisy_x(),
            'y': trajectory.get_noisy_y(),
            't': trajectory.get_time(),
            'DC' : trajectory.directional_correlation_segmentation(window_size=3, threshold=0.3)
        })

        fig1 = px.scatter(df, x="x", y="y", color="DC")

        fig2 = px.line(df, x="x", y="y")
        fig2.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))

        fig = go.Figure(data=fig1.data + fig2.data)
        fig.write_html('file.html')
        """
except KeyboardInterrupt:
    pass

DatabaseHandler.disconnect()
