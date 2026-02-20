"""
All important results like areas, axis lengths, etc. 
are produced within this file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATASET_AS_SAVED = [
    'BTX680R',
    "('Cholesterol and btx', 'BTX680R')",
    'CholesterolPEGKK114',
    "('Cholesterol and btx', 'fPEG-Chol')",
]

states_dataframe = {'immobile': [], 'not-immobile': []}
states_dataframe_average = {'immobile': [], 'not-immobile': []}


plt.figure(figsize=(14, 4))
plt.rcParams.update({'font.size': 19})

colors = ['cyan', 'black', 'gray', 'pink']

values_dict = {}

for state_i, state in enumerate(['directed','normal','subdifussive','confinement']):
    values = []

    df = pd.read_excel(f'Results/not-immobile_{state}-d-i_gs_deepspt_portions_individual_scsa_scda.xlsx')

    for column in df.columns:
        values.extend(df[column].dropna().values.tolist())
    
    values_dict[state]=values
    bins = np.arange(0,0.075,0.005)
    mean_val = np.mean(values)
    std_val = np.std(values)

    plt.subplot(1,4,state_i+1)
    plt.title(state.capitalize())
    plt.hist(values, bins=bins, density=True, color=colors[state_i])
    plt.ylabel("Frequency")
    plt.xlabel(rf"$\overline{'{d_{i}}'}$")
    plt.yticks([])

    plt.text(
        0.95, 0.95,
        rf"$\overline{'{d_{i}}'} = {mean_val:.3f} \mu m$"+"\n"+f"$\sigma = {std_val:.3f} \mu m$",
        ha='right', va='top',
        transform=plt.gca().transAxes
    )

plt.tight_layout()
plt.savefig('scsa graph.svg')

dir_nor = values_dict['directed']+values_dict['normal']
sub_con = values_dict['subdifussive']+values_dict['confinement']

print("Dir+Normal", np.mean(dir_nor), np.std(dir_nor))
print("Sub+Con", np.mean(sub_con), np.std(sub_con))

print(stats.kstest(dir_nor, sub_con))