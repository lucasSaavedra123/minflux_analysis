import pandas as pd
import glob


parameter = 'k'

for a in glob.glob('./Results/*_basic_information.xlsx'):
    df = pd.read_excel(a, parameter)
    print(a, df[parameter].mean(), df[parameter].sem())

#if parameter == 'betha'