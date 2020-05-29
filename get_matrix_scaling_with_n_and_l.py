import numpy as np
from python_utils import file_utils
import os
import pandas as pd

def get_n_l_from_tmpstructures_dir(tmpstructures_dir):
    split_tmpstructures_dir = tmpstructures_dir.split('/')
    for i in range(len(split_tmpstructures_dir)):
        if 'zeta' in split_tmpstructures_dir[i]:
            param_path = split_tmpstructures_dir[i]
            break
    n = int(param_path.split('-')[0].split('n')[-1])
    l = int(param_path.split('-')[1].split('l')[-1])
    return n, l

def main():
    working_dir = os.getcwd()
    soap_runs_dir = os.path.join(working_dir, 'soap_runs')
    tmpstructures_dirs = file_utils.find(soap_runs_dir, 'tmpstructures')
    scaling_data = []
    for tmpstructures_dir in tmpstructures_dirs:
        n, l = get_n_l_from_tmpstructures_dir(tmpstructures_dir)
        structures_files = file_utils.find(tmpstructures_dir, '*.npy')
        soap_matrix = np.load(structures_files[0])
        scaling_data.append([n, l, soap_matrix.shape[1]])

    df = pd.DataFrame(scaling_data, columns=['n', 'l', 'num_columns'])
    df.to_csv(os.path.join(working_dir, 'n_l_scaling.csv'))

main()