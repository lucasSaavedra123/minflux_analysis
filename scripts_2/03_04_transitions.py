"""
All important results like areas, axis lengths, etc. 
are produced within this file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DatabaseHandler import DatabaseHandler
from CONSTANTS import *
from utils import *

def calculate_transition_matrix(secuencia, estados_posibles=[0, 1, 2, 3, 4]):
    n = len(estados_posibles)
    matriz = np.zeros((n, n), dtype=int)
    estado_to_idx = {estado: idx for idx, estado in enumerate(estados_posibles)}

    for i in range(len(secuencia) - 1):
        from_state = secuencia[i]
        to_state = secuencia[i + 1]
        if from_state in estado_to_idx and to_state in estado_to_idx:
            matriz[estado_to_idx[from_state], estado_to_idx[to_state]] += 1

    # Normalizamos por fila
    matriz_prob = matriz.astype(float)
    row_sums = matriz_prob.sum(axis=1, keepdims=True)
    matriz_prob = np.divide(matriz_prob, row_sums, out=np.zeros_like(matriz_prob), where=row_sums != 0)

    return pd.DataFrame(matriz_prob, index=estados_posibles, columns=estados_posibles)

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

TO_SAVE_DATASET = [
    'Control',
    'CDx',
    'BTX680R',
    "('Cholesterol and btx', 'BTX680R')",
    'CholesterolPEGKK114',
    "('Cholesterol and btx', 'fPEG-Chol')",
    'BTX640-CHOL-50-nM',
    "('BTX680-fPEG-CHOL-50-nM', 'BTX680R')",
    "('BTX680-fPEG-CHOL-100-nM', 'BTX680R')",
    "('BTX680-fPEG-CHOL-50-nM', 'fPEG-Chol')",
    "('BTX680-fPEG-CHOL-100-nM', 'fPEG-Chol')",
    'CK666-BTX680',
    "('CK666-BTX680-CHOL', 'BTX680R')",
    'CK666-CHOL',
    "('CK666-BTX680-CHOL', 'fPEG-Chol')",
]

matrices = defaultdict(lambda: [])

for index, dataset in enumerate(new_datasets_list):
    print(dataset,index)

    for dgn_state in range(5):
        basic_query_dict = {'info.dataset': dataset} if index < len(INDIVIDUAL_DATASETS) else {'info.dataset': dataset[0], 'info.classified_experimental_condition':dataset[1]}

        filter_query = basic_query_dict.copy()
        filter_query.update({
            'info.immobile': False,
            f'info.analysis.eco_states':{'$exists':1},
        })

        projection = {'info.file':1, 'info.roi':1,f'info.analysis.eco_states':1}

        documents = list(Trajectory._get_collection().find(filter_query, projection))

        for doc in documents:
            matrices[str(dataset)].append(calculate_transition_matrix(doc['info']['analysis']['eco_states']))

DatabaseHandler.disconnect()

# Promediar las matrices para cada dataset
promedios = {}

for dataset_str, lista_matrices in matrices.items():
    if not lista_matrices:
        continue

    matrices_array = np.stack([df.to_numpy() for df in lista_matrices])
    promedio = np.nanmean(matrices_array, axis=0)
    estados = [0, 1, 2, 3, 4]
    promedio_df = pd.DataFrame(promedio, index=estados, columns=estados)
    promedios[dataset_str] = promedio_df
    promedio_df.to_excel(f'Results/promedio_matriz_transicion_{dataset_str}.xlsx')
