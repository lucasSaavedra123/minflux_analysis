"""
Turning angles files are produced with this script
"""
import pandas as pd
from bson.objectid import ObjectId

from DatabaseHandler import DatabaseHandler
from Trajectory import Trajectory
from CONSTANTS import *
from utils import *


def pack_percentage_values(info):
    dataframe = pd.DataFrame(info)
    dataframe = dataframe.groupby(['file', 'roi']).sum()
    dataframe['All Sum'] = dataframe['Subdiffusive'] + dataframe['Brownian'] + dataframe['Superdiffusive']
    dataframe['Subdiffusive']/=dataframe['All Sum']
    dataframe['Subdiffusive'] *= 100
    dataframe['Brownian']/=dataframe['All Sum']
    dataframe['Brownian'] *= 100
    dataframe['Superdiffusive']/=dataframe['All Sum']
    dataframe['Superdiffusive'] *= 100

    dataframe['Subdiffusive_mean'] = dataframe['Subdiffusive'].mean()
    dataframe['Subdiffusive_sem'] = dataframe['Subdiffusive'].sem()

    dataframe['Brownian_mean'] = dataframe['Brownian'].mean()
    dataframe['Brownian_sem'] = dataframe['Brownian'].sem()

    dataframe['Superdiffusive_mean'] = dataframe['Superdiffusive'].mean()
    dataframe['Superdiffusive_sem'] = dataframe['Superdiffusive'].sem()

    return dataframe

APPLY_GS_CRITERIA = True

DatabaseHandler.connect_over_network(None, None, IP_ADDRESS, COLLECTION_NAME)

INDIVIDUAL_DATASETS = [
    'Control',
    'CDx',
    'BTX680R',
    'CholesterolPEGKK114',
    'CK666-BTX680',
    'CK666-CHOL',
    'BTX640-CHOL-50-nM',
    'BTX640-CHOL-50-nM-LOW-DENSITY',
]

new_datasets_list = INDIVIDUAL_DATASETS.copy()

for combined_dataset in [
    'Cholesterol and btx',
    'CK666-BTX680-CHOL',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]:
    new_datasets_list.append((combined_dataset, BTX_NOMENCLATURE))
    new_datasets_list.append((combined_dataset, CHOL_NOMENCLATURE))

for index, dataset in enumerate(new_datasets_list):
    print(dataset)
    filter_query = {'info.dataset': dataset, 'info.immobile': False} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1], 'info.immobile': False}

    all_bethas_infos = []
    all_confined_bethas_infos = []
    all_non_confined_bethas_infos = []

    with pd.ExcelWriter(f"./Results/{dataset}_{index}_gs_{APPLY_GS_CRITERIA}_anomalous_classification_information.xlsx") as writer:        
        for label in DIFFUSION_BEHAVIOURS_INFORMATION:
            ids = get_ids_of_trayectories_under_betha_limits(
                filter_query,
                DIFFUSION_BEHAVIOURS_INFORMATION[label]['range_0'],
                DIFFUSION_BEHAVIOURS_INFORMATION[label]['range_1'],
            )

            bethas_infos = Trajectory._get_collection().find(
                {'_id': {'$in':[ObjectId(an_id) for an_id in ids]}},
                {f'info.analysis.goodness_of_fit':1, f'info.analysis.confinement-betha':1, f'info.analysis.non-confinement-betha':1, f'info.analysis.confinement-goodness_of_fit':1, f'info.analysis.non-confinement-goodness_of_fit':1, f'info.roi':1, f'info.file': 1}
            )

            for info in bethas_infos:
                if info['info']['analysis']['goodness_of_fit'] > 0.8:
                    all_bethas_infos.append({
                        'file': info['info']['file'],
                        'roi': info['info']['roi'],
                        'Subdiffusive': 0,
                        'Brownian': 0,
                        'Superdiffusive': 0
                    })

                    all_bethas_infos[-1][label] = 1

                for aux_label in DIFFUSION_BEHAVIOURS_INFORMATION:
                    for confined_betha, confined_gof in zip(info['info']['analysis']['confinement-betha'], info['info']['analysis']['confinement-goodness_of_fit']):
                        if confined_betha is not None and confined_gof > 0.8 and DIFFUSION_BEHAVIOURS_INFORMATION[aux_label]['range_0'] < confined_betha < DIFFUSION_BEHAVIOURS_INFORMATION[aux_label]['range_1']:
                            all_confined_bethas_infos.append({
                                'file': info['info']['file'],
                                'roi': info['info']['roi'],
                                'Subdiffusive': 0,
                                'Brownian': 0,
                                'Superdiffusive': 0
                            })

                            all_confined_bethas_infos[-1][aux_label] = 1
                    for non_confined_betha, non_confined_gof in zip(info['info']['analysis']['non-confinement-betha'], info['info']['analysis']['non-confinement-goodness_of_fit']):
                        if non_confined_betha is not None and non_confined_gof > 0.8 and DIFFUSION_BEHAVIOURS_INFORMATION[aux_label]['range_0'] < non_confined_betha < DIFFUSION_BEHAVIOURS_INFORMATION[aux_label]['range_1']:
                            all_non_confined_bethas_infos.append({
                                'file': info['info']['file'],
                                'roi': info['info']['roi'],
                                'Subdiffusive': 0,
                                'Brownian': 0,
                                'Superdiffusive': 0
                            })

                            all_non_confined_bethas_infos[-1][aux_label] = 1

        pack_percentage_values(all_bethas_infos).to_excel(writer, sheet_name='Classification', index=False)
        pack_percentage_values(all_non_confined_bethas_infos).to_excel(writer, sheet_name='Non Confinement Classification', index=False)
        pack_percentage_values(all_confined_bethas_infos).to_excel(writer, sheet_name='Confinement Classification', index=False)

DatabaseHandler.disconnect()
