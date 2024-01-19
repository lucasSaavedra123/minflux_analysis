"""
Residence times are plotted
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from CONSTANTS import *
from utils import *
from scipy.stats import expon


def plot_dataset(file, color):
    residence_times = []

    residence_times += pd.read_excel(open(os.path.join('Results',file), 'rb'), sheet_name='residence_time')['residence_time'].tolist()

    #Exponential Fitting
    loc, scale = expon.fit(residence_times, floc=0)

    x = np.arange(0.01,4,0.001)
    pdfs = expon.pdf(x, loc=0, scale=scale)

    plt.plot(x,pdfs, color=color, linewidth=2.5)
    plt.hist(residence_times, density=True, bins='auto', histtype='stepfilled', alpha=0.2, color=color)

def do_plot(xlim=[0,2], ylim=[0,3]):
    plt.xlabel('Confinement Time [s]', fontname="Arial", fontsize=30)
    plt.yticks(visible=False)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks([0,1,2])
    plt.xticks(fontname="Arial", fontsize=30)
    plt.ylabel('Frequency', fontname="Arial", fontsize=30)
    plt.tight_layout()
    plt.show()


plot_dataset('Control_0_basic_information.xlsx', 'red')
plot_dataset('CDx_1_basic_information.xlsx', 'blue')
do_plot([0,2], [0,2])

plot_dataset('BTX680R_2_basic_information.xlsx', 'green')
plot_dataset('BTX680R_4_basic_information.xlsx', '#ff0090')
do_plot([0,2], [0,2])

plot_dataset('CholesterolPEGKK114_3_basic_information.xlsx', 'orange')
plot_dataset('fPEG-Chol_5_basic_information.xlsx', 'black')
do_plot([0,2], [0,3])