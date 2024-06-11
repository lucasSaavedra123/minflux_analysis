IP_ADDRESS = 'localhost'
COLLECTION_NAME = 'MINFLUX_DATA'
FILE_AND_ROI_FILE_CACHE = 'file_and_roi.txt'

DELTA_T = 0.000132
TIME_START = 0.000084
DIMENSION = 2
R = 1/6
SEGMENT_LENGTH = 500
MAX_T = 0.050
SUB_MAX_T = 0.025

DATASETS_LIST = [
    'Control',
    'CDx',
    'BTX680R',
    'CholesterolPEGKK114',
    'Cholesterol and btx',
    'CK666-BTX680',
    'CK666-CHOL',
    'CK666-BTX680-CHOL',
    'BTX640-CHOL-50-nM',
    'BTX640-CHOL-50-nM-LOW-DENSITY',
    'BTX680-fPEG-CHOL-50-nM',
    'BTX680-fPEG-CHOL-100-nM',
]

DATASET_TO_COLOR = {
   DATASETS_LIST[0]: 'blue',
   DATASETS_LIST[1]: 'red',
   DATASETS_LIST[2]: 'orange',
   DATASETS_LIST[3]: 'green',
   DATASETS_LIST[4]: 'purple',
}

"""
DIFFUSION_BEHAVIOURS_INFORMATION = {
    'Subdiffusive I': {'range_0': float('-inf'), 'range_1': 0.5},
    'Subdiffusive II': {'range_0': 0.5,'range_1': 0.7},
    'Subdiffusive III': {'range_0': 0.7, 'range_1': 0.9},
    'Brownian': {'range_0': 0.9,'range_1': 1.1},
    'Superdiffusive': {'range_0': 1.1,'range_1': float('inf')},
}
"""
DIFFUSION_BEHAVIOURS_INFORMATION = {
    'Subdiffusive': {'range_0': float('-inf'), 'range_1': 0.9},
    'Brownian': {'range_0': 0.9,'range_1': 1.1},
    'Superdiffusive': {'range_0': 1.1,'range_1': float('inf')},
}

NUMBER_OF_POINTS_FOR_MSD = 250
GOODNESS_OF_FIT_MAXIMUM = float('inf')
STEP_LAGS_FOR_ANGLE_ANALYSIS = [1,4,8,25,50]

TDCR_THRESHOLD = 0.55
GS_THRESHOLD = 5.121

BTX_NOMENCLATURE = 'BTX680R'
CHOL_NOMENCLATURE = 'fPEG-Chol'

def default_angles():
    return {str(step_lag): [] for step_lag in STEP_LAGS_FOR_ANGLE_ANALYSIS}

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def batch_for_gen(generator, n=1):
    return_to_list = []
    
    for element in generator:
        if len(return_to_list) == n:
            yield return_to_list
            return_to_list = [element]
        else:
            return_to_list.append(element)
    
    yield return_to_list
