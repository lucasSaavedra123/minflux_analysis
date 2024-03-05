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

def analyze_dcr(datasets_one, datasets_two, datasets_combined, dataset_one_label, dataset_two_label):
    plt.rcParams['savefig.dpi'] = 500
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    btx_dcr_values = []
    chol_dcr_values = []

    for dataset_one in datasets_one:
        btx_dcr_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_one}, 'dcr')]

    for dataset_two in datasets_two:
        chol_dcr_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_two}, 'dcr')]

    dcr_dataframe = pd.DataFrame({
        'Track detection channel ratio (tDCR)': btx_dcr_values + chol_dcr_values,
        'Experimental condition': [dataset_one_label] * len(btx_dcr_values) + [dataset_two_label] * len(chol_dcr_values)
    })

    DatabaseHandler.disconnect()

    sns.set(font_scale=3.5)
    sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', hue='Experimental condition', kde=True)
    plt.show()

    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    chol_and_btx_dcr_values = []

    for dataset_combined in datasets_combined:
        chol_and_btx_dcr_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_combined}, 'dcr')]

    dcr_dataframe = pd.DataFrame({
        'Track detection channel ratio (tDCR)': chol_and_btx_dcr_values,
        'Experimental condition': [f'{dataset_two_label}(+{dataset_one_label})' + "\n+\n" + f'{dataset_one_label}(+{dataset_two_label})'] * len(chol_and_btx_dcr_values)
    })

    DatabaseHandler.disconnect()

    sns.set(font_scale=3.5)
    sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', color='#805380', edgecolor='#634163', kde=True)
    plt.show()

btx_alone_datasets = ['BTX680R', 'CK666-BTX680']
chol_alone_datasets = ['CholesterolPEGKK114', 'CK666-CHOL']
btx_and_chol_datasets = ['BTX680-fPEG-CHOL-50-nM', 'BTX680-fPEG-CHOL-100-nM', 'Cholesterol and btx', 'CK666-BTX680-CHOL']

analyze_dcr(btx_alone_datasets, chol_alone_datasets, btx_and_chol_datasets, r'CF$^{®}$680R-BTX', 'fPEG-Chol')
