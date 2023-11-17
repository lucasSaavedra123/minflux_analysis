import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from utils import *


APPLY_GS_CRITERIA = True

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

btx_dcr_values = get_list_of_values_of_field('BTX680R', 'dcr', False)
btx_dcr_values = [np.mean(dcr_values) for dcr_values in btx_dcr_values]

chol_dcr_values = get_list_of_values_of_field('CholesterolPEGKK114', 'dcr', False)
chol_dcr_values = [np.mean(dcr_values) for dcr_values in chol_dcr_values]

dcr_dataframe = pd.DataFrame({
    'Track detection channel ratio (tDCR)': btx_dcr_values + chol_dcr_values,
    'Experimental condition': ['BTX680R'] * len(btx_dcr_values) + ['fPEG-Chol'] * len(chol_dcr_values)
})

DatabaseHandler.disconnect()

sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', hue='Experimental condition', kde=True)
plt.show()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

chol_and_btx_dcr_values = get_list_of_values_of_field('Cholesterol and btx', 'dcr', False)
chol_and_btx_dcr_values = [np.mean(dcr_values) for dcr_values in chol_and_btx_dcr_values]

dcr_dataframe = pd.DataFrame({
    'Track detection channel ratio (tDCR)': chol_and_btx_dcr_values,
    'Experimental condition': ['fPEG-Chol+BTX680R'] * len(chol_and_btx_dcr_values)
})

DatabaseHandler.disconnect()

sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', hue='Experimental condition', kde=True)
plt.show()
