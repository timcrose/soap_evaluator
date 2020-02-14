from python_utils import file_utils, list_utils, math_utils
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def get_num_structs_to_use(soap_owd):
    '''
    soap_owd: str
        Original working dir (submission dir with soap.conf) of a soap run.
    
    Return: int
        num_structures_to_use for the given soap owd.
    
    Purpose: Parse soap.conf in soap_owd for num_structures_to_use

    TODO: Make more robust by searching kernel size or other places the number
        of structures is output in output files.
    '''
    soap_conf_fpath = file_utils.os.path.join(soap_owd, 'soap.conf')
    print('Parsing', soap_conf_fpath, 'for num_structures_to_use', flush=True)
    found_lines = file_utils.grep('num_structures_to_use', soap_conf_fpath, read_mode='r', fail_if_DNE=True)
    if len(found_lines) > 0:
        found_line = found_lines[0]
    else:
        return -55555.0
    # format: num_structures_to_use = 100
    return int(found_line.split('=')[1])


def get_num_cores(soap_owd):
    '''
    soap_owd: str
        Original working dir (submission dir with soap.conf) of a soap run.
    
    Return: int
        number of MPI ranks (cores) used for the given soap owd.
    
    Purpose: Parse output.out in soap_owd for number of MPI ranks used.
    '''
    output_fpath = file_utils.os.path.join(soap_owd, 'output.out')
    print('Parsing', output_fpath, 'for number of MPI ranks', flush=True)
    found_lines = file_utils.grep(' MPI ranks on ', output_fpath, read_mode='r', fail_if_DNE=True)
    if len(found_lines) > 0:
        found_line = found_lines[0]
    else:
        return -55555.0
    #print(int(found_line.split()[1]))
    # format: ('using 2 MPI ranks on 1200 structures.',)
    return int(found_line.split()[1])


def get_param_values(param_path):
    param_basename = file_utils.os.path.basename(param_path)
    #format n8-l8-c10.0-g0.7-zeta1.0
    n = int(param_basename.split('-')[0].split('n')[-1])
    l = int(param_basename.split('-')[1].split('l')[-1])
    c = float(param_basename.split('-')[2].split('c')[-1])
    g = float(param_basename.split('-')[3].split('g')[-1])
    zeta = float(param_basename.split('-')[4].split('zeta')[-1])
    return n, l, c, g, zeta


def get_avg_time(param_path, data_key):
    '''
    param_path: str
        Parameter path under soap_runs directory.

    data_key: str
        Key that determines which type of data to extract.

    Return:
    avg_time: float
        Average time taken by all ranks for this param_path in soap_owd for
        calculating the property / quantity given by data_key. np.nan returned if 
        data_key not found in output file.

    Purpose: Parse output log files for search strings corresponding to data_key and 
        get the average for all MPI ranks in a particular param_path.
    '''
    search_str_dct = {'total_time' : 'total time', 'kernel_time' : 'time to calculate kernel',
                        'env_time' : 'calculate environments', 'param_time': 'time for param'}
    outfiles = file_utils.find(param_path, 'output_from_rank*')
    #print('outfiles', outfiles, flush=True)
    found_lines = file_utils.grep(search_str_dct[data_key], outfiles, verbose=True)
    #print('found_lines', found_lines, flush=True)
    time_lst = []
    for found_line in found_lines:
        if data_key == 'total_time' or data_key == 'kernel_time' or data_key == 'env_time' or data_key == 'param_time':
            # format: ('total time', 156.7616991996765)
            # format: ('time to calculate kernel', 124.95694470405579)
            # format: ('time to read input structures and calculate environments', 11.158977270126343)
            time_lst.append(float(found_line.split()[-1].split(')')[0]))
    if len(time_lst) > 0:
        avg_time = sum(time_lst) / float(len(time_lst))
    else:
        avg_time = -55555.0
    return avg_time


def get_data(data_to_get, soap_owds, num_decimal_places=1, sort_by=None):
    '''
    data_to_get: list of str
        List of data to get. The order specified is the order in which
        they will appear in columns of a csv file later. Valid options
        so far are: 'num_structures_to_use', 'total_time'

    soap_owds: list of str
        List of original working dir (submission dir with soap.conf) for each
        soap owd to include in the data file.

    num_decimal_places: int
        Number of decimal places to round each number

    sort_by: str or None
        str: Sort data_matrix by the column represented by this argument.
        None: Do not sort data_matrix

    Return: list of list of float
        Each column is a parameter specified by data_to_get. Each row is for each
        param_path in each soap owd provided by soap_owds.

    Purpose: Get the data from soap output files that can be written to a data file.
    '''
    #data_keys_with_data_per_param_path = {'total_time', 'kernel_time', 'env_time'}
    #num_data_keys_with_data_per_param_path = len(data_keys_with_data_per_param_path)
    #data_per_param_path = len(set(data_to_get) - data_keys_with_data_per_param_path) < num_data_keys_with_data_per_param_path
    data_matrix = None
    for soap_owd in soap_owds:
        param_paths = file_utils.find(soap_owd, '*zeta*')
        data_bit = -55555.0 * np.ones((len(param_paths), len(data_to_get)))
        for i, param_path in enumerate(param_paths):
            n, l, c, g, zeta = get_param_values(param_path)
            for j, data_key in enumerate(data_to_get):
                if data_key == 'num_structures_to_use':
                    data_bit[i,j] = math_utils.round(get_num_structs_to_use(soap_owd), num_decimal_places)
                elif data_key == 'num_cores':
                    data_bit[i,j] = math_utils.round(get_num_cores(soap_owd), num_decimal_places)
                elif data_key == 'total_time' or data_key == 'kernel_time' or data_key == 'env_time' or data_key == 'param_time':
                    data_bit[i, j] = math_utils.round(get_avg_time(param_path, data_key), num_decimal_places)
                elif data_key == 'n':
                    data_bit[i,j] = n
                elif data_key == 'l':
                    data_bit[i,j] = l
                elif data_key == 'g':
                    data_bit[i,j] = g
                elif data_key == 'c':
                    data_bit[i,j] = c
                elif data_key == 'zeta':
                    data_bit[i,j] = zeta
            
        if data_matrix is None:
            data_matrix = deepcopy(data_bit)
        else:
            data_matrix = np.vstack((data_matrix, data_bit))
        
    if isinstance(sort_by, str) and sort_by in data_to_get:
        data_matrix = list_utils.sort_by_col(data_matrix, data_to_get.index(sort_by))
    return list(map(list, data_matrix))


def plot_data(data_matrix, data_to_get):
    y_var = 'kernel_time'
    x_var = 'num_cores'
    append_to_title = '_target1_2mpc_at_n-8,l-8,c4.0,g-0.7'

    col_dct = {d:data_to_get.index(d) for d in data_to_get}
    data_matrix = np.array(data_matrix)
    axis_label_dct = {'n':'number of radial basis functions',
                      'l':'number of angular basis functions',
                      'c':'cutoff radius',
                      'R^2':'Test set R^2',
                      'param_time': 'Total time taken for the workflow of this parameter set (s)',
                      'num_cores': 'Number of MPI ranks',
                      'kernel_time': 'Time to construct the kernel'}
    plot_title = y_var + '_vs_' + x_var + append_to_title
    y_axis_label = axis_label_dct[y_var]
    x_axis_label = axis_label_dct[x_var]

    f = plt.figure()
    plt.title(plot_title)
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.plot(data_matrix[:,col_dct[x_var]], data_matrix[:,col_dct[y_var]], '-o')
    f.savefig(plot_title + '.png', bbox_inches='tight')


def main():
    outfile_fpath = 'benchmark_data.csv'
    data_to_get = ['param_time', 'num_cores', 'kernel_time', 'env_time', 'n', 'l', 'c', 'g', 'zeta']
    sort_by = 'num_cores'
    num_decimal_places = 1
    soap_owds = file_utils.find(file_utils.os.getcwd(), 'num_cores*')
    #soap_owds = ['/global/cscratch1/sd/trose/soap_run_calcs/target1/2mpc/hyperparameter_optimization/cutoff']
    print('soap_owds', soap_owds, flush=True)
    data_matrix = get_data(data_to_get, soap_owds, num_decimal_places=num_decimal_places, sort_by=sort_by)

    plot_data(data_matrix, data_to_get)

    data_matrix = [data_to_get] + data_matrix
    print('data_matrix', flush=True)
    print(data_matrix, flush=True)
    file_utils.rm(outfile_fpath)
    file_utils.write_rows_to_csv(outfile_fpath, data_matrix)

if __name__ == '__main__':
    main()