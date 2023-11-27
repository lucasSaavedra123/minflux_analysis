import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon, sem
from scipy.spatial.distance import pdist
from bson.objectid import ObjectId

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *


APPLY_GS_CRITERIA = True

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)


new_datasets_list = DATASETS_LIST.copy()
new_datasets_list = DATASETS_LIST[:-1]
new_datasets_list.append(BTX_NOMENCLATURE)
new_datasets_list.append(CHOL_NOMENCLATURE)

#TODO

DatabaseHandler.disconnect()