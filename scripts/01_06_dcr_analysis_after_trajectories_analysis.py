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

def analyze_dcr(datasets_one, datasets_two, datasets_combined, dataset_one_label, dataset_two_label, file_name=None):
    plt.rcParams['savefig.dpi'] = 600
    DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

    if not(len(datasets_one) == 0 and len(datasets_two) == 0):
        dataset_one_dcr_values = []
        dataset_two_dcr_values = []

        for dataset_one in datasets_one:
            dataset_one_dcr_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_one, 'info.immobile': False}, 'dcr')]

        for dataset_two in datasets_two:
            dataset_two_dcr_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_two, 'info.immobile': False}, 'dcr')]

        dcr_dataframe = pd.DataFrame({
            'Track detection channel ratio (tDCR)': dataset_one_dcr_values + dataset_two_dcr_values,
            'Experimental condition': [dataset_one_label] * len(dataset_one_dcr_values) + [dataset_two_label] * len(dataset_two_dcr_values)
        })

        sns.set(font_scale=2)
        sns.set_style("whitegrid")
        sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', hue='Experimental condition', kde=True, legend=False, binwidth=0.01, stat='probability')
        plt.ylabel('Fraction')
        plt.xlim([0,1])
        plt.ylim([0,0.3])
        plt.xticks([0.2,0.4,0.6,0.8,1.0])
        plt.axvline(TDCR_THRESHOLD, color='black', linestyle='--', linewidth=2)
        plt.tight_layout()

        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name+'_separated.svg')
    plt.clf()
    if len(datasets_combined) != 0:
        datasets_combined_dcr_values = []

        for dataset_combined in datasets_combined:
            datasets_combined_dcr_values += [np.mean(dcr_values) for dcr_values in get_list_of_values_of_field({'info.dataset': dataset_combined, 'info.immobile': False}, 'dcr')]

        dcr_dataframe = pd.DataFrame({
            'Track detection channel ratio (tDCR)': datasets_combined_dcr_values,
            'Experimental condition': [f'{dataset_two_label}(+{dataset_one_label})' + "\n+\n" + f'{dataset_one_label}(+{dataset_two_label})'] * len(datasets_combined_dcr_values)
        })

        sns.set(font_scale=2)
        sns.set_style("whitegrid")
        sns.histplot(data=dcr_dataframe, x='Track detection channel ratio (tDCR)', color='#805380', edgecolor='#634163', kde=True, legend=False, binwidth=0.01, stat='probability')
        plt.ylabel('Fraction')
        plt.xlim([0,1])
        plt.ylim([0,0.3])
        plt.xticks([0.2,0.4,0.6,0.8,1.0])
        plt.axvline(TDCR_THRESHOLD, color='black', linestyle='--', linewidth=2)
        plt.tight_layout()

        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name+'_combined.svg')
    plt.clf()
    DatabaseHandler.disconnect()

# btx_alone_datasets = ['BTX680R', 'CK666-BTX680']
# chol_alone_datasets = ['CholesterolPEGKK114', 'CK666-CHOL']
# btx_and_chol_datasets = ['BTX680-fPEG-CHOL-50-nM', 'BTX680-fPEG-CHOL-100-nM', 'Cholesterol and btx', 'CK666-BTX680-CHOL']

# analyze_dcr(btx_alone_datasets, chol_alone_datasets, btx_and_chol_datasets, r'CF$^{®}$680R-BTX', 'fPEG-Chol')

btx_alone_datasets = ['BTX680R']
chol_alone_datasets = ['CholesterolPEGKK114']
btx_and_chol_datasets = ['Cholesterol and btx']

analyze_dcr(btx_alone_datasets, chol_alone_datasets, btx_and_chol_datasets, r'CF$^{®}$680R-BTX', 'fPEG-Chol', 'DCR-HISTOGRAM-No-CK666')

btx_alone_datasets = ['CK666-BTX680']
chol_alone_datasets = ['CK666-CHOL']
btx_and_chol_datasets = ['CK666-BTX680-CHOL']

analyze_dcr(btx_alone_datasets, chol_alone_datasets, btx_and_chol_datasets, r'CF$^{®}$680R-BTX', 'fPEG-Chol', 'DCR-HISTOGRAM-With-CK666')

btx_and_chol_datasets = ['BTX680-fPEG-CHOL-50-nM']
analyze_dcr([], [], btx_and_chol_datasets, r'CF$^{®}$680R-BTX', 'fPEG-Chol', 'DCR-HISTOGRAM-50nM')

btx_and_chol_datasets = ['BTX680-fPEG-CHOL-100-nM']
analyze_dcr([], [], btx_and_chol_datasets, r'CF$^{®}$680R-BTX', 'fPEG-Chol', 'DCR-HISTOGRAM-100nM')