import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *


DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

localization_precisions = [r.get('info',{}).get('analysis',{}).get('localization_precision') for r in Trajectory._get_collection().find({}, {'_id':1, 'info.analysis.localization_precision':1})]
localization_precisions = [v*1000 for v in localization_precisions if v is not None and v*1000 > 1]

print(np.mean(localization_precisions), sem(localization_precisions))

plt.hist(localization_precisions, bins=100)
plt.show()

DatabaseHandler.disconnect()
