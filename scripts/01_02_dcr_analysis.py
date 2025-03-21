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

    dataset_one_dcr_values, dataset_one_efo_values = [], []
    dataset_two_dcr_values, dataset_two_efo_values = [], []

    for dataset_one in datasets_one:
        dataset_one_dcr_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_one}, 'dcr')]
        dataset_one_efo_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_one}, 'intensity')]

    for dataset_two in datasets_two:
        dataset_two_dcr_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_two}, 'dcr')]
        dataset_two_efo_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_two}, 'intensity')]

    dcr_dataframe = pd.DataFrame({
        'EFO': dataset_one_efo_values + dataset_two_efo_values,
        'Track detection channel ratio (tDCR)': dataset_one_dcr_values + dataset_two_dcr_values,
        'Experimental condition': [dataset_one_label] * len(dataset_one_dcr_values) + [dataset_two_label] * len(dataset_two_dcr_values)
    })

    sns.set(font_scale=3.5)
    sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', hue='Experimental condition', kde=True)
    plt.xlim([0,1])
    plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.show()

    sns.set(font_scale=3.5)
    sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', y='EFO', hue='Experimental condition', kde=True)
    plt.xlim([0,1])
    plt.ylim([0,1e6])
    plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.show()

    datasets_combined_dcr_values, datasets_combined_efo_values = [], []

    for dataset_combined in datasets_combined:
        datasets_combined_dcr_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_combined}, 'dcr')]
        datasets_combined_efo_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_combined}, 'intensity')]

    dcr_dataframe = pd.DataFrame({
        'EFO': datasets_combined_efo_values,
        'Track detection channel ratio (tDCR)': datasets_combined_dcr_values,
        'Experimental condition': [f'{dataset_two_label}(+{dataset_one_label})' + "\n+\n" + f'{dataset_one_label}(+{dataset_two_label})'] * len(datasets_combined_dcr_values)
    })

    sns.set(font_scale=3.5)
    sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', color='#805380', edgecolor='#634163', kde=True)
    plt.xlim([0,1])
    plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.show()

    sns.set(font_scale=3.5)
    sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', y='EFO', color='#805380', kde=True)
    plt.xlim([0,1])
    plt.ylim([0,1e6])
    plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.show()

    DatabaseHandler.disconnect()

btx_alone_datasets = ['BTX680R', 'CK666-BTX680']
chol_alone_datasets = ['CholesterolPEGKK114', 'CK666-CHOL']
btx_and_chol_datasets = ['BTX680-fPEG-CHOL-50-nM', 'BTX680-fPEG-CHOL-100-nM', 'Cholesterol and btx', 'CK666-BTX680-CHOL']

analyze_dcr(btx_alone_datasets, chol_alone_datasets, btx_and_chol_datasets, r'CF$^{®}$680R-BTX', 'fPEG-Chol')
