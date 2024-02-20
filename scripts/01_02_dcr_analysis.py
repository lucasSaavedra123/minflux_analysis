"""
Some experiments included at the same time Chol and BTX.
Hence, to classify trajectories between both types,
this script analyze the suitable DCR threshold to 
accomplish the classification.  
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from utils import *

def analyze_dcr(dataset_one, dataset_two, dataset_combined, dataset_one_label, dataset_two_label):
    plt.rcParams['savefig.dpi'] = 500
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    btx_dcr_values = get_list_of_values_of_field({'info.dataset': dataset_one}, 'dcr')
    btx_dcr_values = [np.mean(dcr_values) for dcr_values in btx_dcr_values]

    chol_dcr_values = get_list_of_values_of_field({'info.dataset': dataset_two}, 'dcr')
    chol_dcr_values = [np.mean(dcr_values) for dcr_values in chol_dcr_values]

    dcr_dataframe = pd.DataFrame({
        'Track detection channel ratio (tDCR)': btx_dcr_values + chol_dcr_values,
        'Experimental condition': [dataset_one_label] * len(btx_dcr_values) + [dataset_two_label] * len(chol_dcr_values)
    })

    DatabaseHandler.disconnect()

    sns.set(font_scale=3.5)
    sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', hue='Experimental condition', kde=True)
    plt.show()

    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    chol_and_btx_dcr_values = get_list_of_values_of_field({'info.dataset': dataset_combined}, 'dcr')
    chol_and_btx_dcr_values = [np.mean(dcr_values) for dcr_values in chol_and_btx_dcr_values]

    dcr_dataframe = pd.DataFrame({
        'Track detection channel ratio (tDCR)': chol_and_btx_dcr_values,
        'Experimental condition': [f'{dataset_two_label}(+{dataset_one_label})' + "\n+\n" + f'{dataset_one_label}(+{dataset_two_label})'] * len(chol_and_btx_dcr_values)
    })

    DatabaseHandler.disconnect()

    sns.set(font_scale=3.5)
    sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', color='#805380', edgecolor='#634163', kde=True)
    plt.show()

analyze_dcr('BTX680R', 'CholesterolPEGKK114', 'Cholesterol and btx', r'CF$^{®}$680R-BTX', 'fPEG-Chol')
analyze_dcr('CK666-BTX680', 'CK666-CHOL', 'CK666-BTX680-CHOL', r'CF$^{®}$680R-BTX', 'fPEG-Chol')