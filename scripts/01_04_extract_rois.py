"""
This script upload all trajectories to the MongoDB database.
"""
import os
from collections import defaultdict

from matplotlib import pyplot as plt
import pandas as pd
import tqdm
from roipoly import RoiPoly
from shapely.geometry import Polygon, Point

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, 'MINFLUX_DATA')

files = Trajectory.objects().distinct(field="info.file")

Trajectory._get_collection().update_many({}, {"$unset": {'info.roi':""}})

for file in files:
    roi_index = 0
    q = {'info.file': file}

    p = {
        'x':1,
        'y':1,
        'info.trajectory_id':1
    }
    
    trajectories_infos = list(Trajectory._get_collection().find(q, p))

    raw_dataframe = {
        'x': [],
        'y': [],
        'trajectory_id': []
    }

    for sub_info in trajectories_infos:
        raw_dataframe['x'] += sub_info['x']
        raw_dataframe['y'] += sub_info['y']
        raw_dataframe['trajectory_id'] += [sub_info['info']['trajectory_id']] * len(sub_info['x'])

    dataframe = pd.DataFrame(raw_dataframe)

    while len(dataframe) != 0:
        plt.scatter(dataframe['x'], dataframe['y'], s=1)

        polygon = Polygon(RoiPoly(color='r').get_roi_coordinates())
        dataframe['inside_roi'] = dataframe.apply(lambda row: polygon.contains(Point(row.x, row.y)), axis = 1)
        trajectories_ids = dataframe[dataframe['inside_roi']]['trajectory_id'].unique().tolist()

        q = {
            'info.file':file,
            'info.trajectory_id': {'$in':trajectories_ids}
        }

        p = {
            "$set": {'info.roi': roi_index}
        }

        x = Trajectory._get_collection().update_many(q, p)

        assert x.modified_count == len(trajectories_ids)

        dataframe = dataframe[~dataframe['inside_roi']]
        roi_index += 1

DatabaseHandler.disconnect()
