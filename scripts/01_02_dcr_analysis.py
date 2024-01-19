"""
Some experiments included at the same time Chol and BTX.
Hence, to classify trajectories between both types,
this script analyze the suitable DCR threshold to 
accomplish the classification.  
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from utils import *


plt.rcParams['savefig.dpi'] = 500
DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

btx_dcr_values = get_list_of_values_of_field({'info.dataset': 'BTX680R'}, 'dcr')
btx_dcr_values = [np.mean(dcr_values) for dcr_values in btx_dcr_values]

chol_dcr_values = get_list_of_values_of_field({'info.dataset': 'CholesterolPEGKK114'}, 'dcr')
chol_dcr_values = [np.mean(dcr_values) for dcr_values in chol_dcr_values]

dcr_dataframe = pd.DataFrame({
    'Track detection channel ratio (tDCR)': btx_dcr_values + chol_dcr_values,
    'Experimental condition': [r'CF$^{®}$680R-BTX'] * len(btx_dcr_values) + ['fPEG-Chol'] * len(chol_dcr_values)
})

DatabaseHandler.disconnect()

sns.set(font_scale=3.5)
sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', hue='Experimental condition', kde=True)
plt.show()

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

chol_and_btx_dcr_values = get_list_of_values_of_field({'info.dataset': 'Cholesterol and btx'}, 'dcr')
chol_and_btx_dcr_values = [np.mean(dcr_values) for dcr_values in chol_and_btx_dcr_values]

dcr_dataframe = pd.DataFrame({
    'Track detection channel ratio (tDCR)': chol_and_btx_dcr_values,
    'Experimental condition': [r'fPEG-Chol(+CF$^{®}$680R-BTX)' + "\n+\n" + r'CF$^{®}$680R-BTX(+fPEG-Chol)'] * len(chol_and_btx_dcr_values)
})

DatabaseHandler.disconnect()

sns.set(font_scale=3.5)
sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', color='#805380', edgecolor='#634163', kde=True)
plt.show()
