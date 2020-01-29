from python_utils import file_utils, list_utils, math_utils
import numpy as np
from copy import deepcopy

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
    print('Parsing', soap_conf_fpath, flush=True)
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
    print('Parsing', output_fpath, flush=True)
    found_lines = file_utils.grep(' MPI ranks on ', output_fpath, read_mode='r', fail_if_DNE=True)
    if len(found_lines) > 0:
        found_line = found_lines[0]
    else:
        return -55555.0
    #print(int(found_line.split()[1]))
    # format: ('using 2 MPI ranks on 1200 structures.',)
    return int(found_line.split()[1])


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
                        'env_time' : 'calculate environments'}
    outfiles = file_utils.find(param_path, 'output_from_rank*')
    #print('outfiles', outfiles, flush=True)
    found_lines = file_utils.grep(search_str_dct[data_key], outfiles, verbose=True)
    #print('found_lines', found_lines, flush=True)
    time_lst = []
    for found_line in found_lines:
        if data_key == 'total_time':
            # format: ('total time', 156.7616991996765)
            time_lst.append(float(found_line.split()[-1].split(')')[0]))
        elif data_key == 'kernel_time':
            # format: ('time to calculate kernel', 124.95694470405579)
            time_lst.append(float(found_line.split()[-1].split(')')[0]))
        elif data_key == 'env_time':
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
            for j, data_key in enumerate(data_to_get):
                if data_key == 'num_structures_to_use':
                    data_bit[i,j] = math_utils.round(get_num_structs_to_use(soap_owd), num_decimal_places)
                elif data_key == 'num_cores':
                    data_bit[i,j] = math_utils.round(get_num_cores(soap_owd), num_decimal_places)
                elif data_key == 'total_time' or data_key == 'kernel_time' or data_key == 'env_time':
                    data_bit[i, j] = math_utils.round(get_avg_time(param_path, data_key), num_decimal_places)
            
        if data_matrix is None:
            data_matrix = deepcopy(data_bit)
        else:
            data_matrix = np.vstack((data_matrix, data_bit))
        
    if isinstance(sort_by, str) and sort_by in data_to_get:
        data_matrix = list_utils.sort_by_col(data_matrix, data_to_get.index(sort_by))
    return list(map(list, data_matrix))


def main():
    outfile_fpath = 'num_cores_benchmark_data.csv'
    data_to_get = ['num_cores', 'kernel_time', 'env_time', 'total_time']
    sort_by = 'num_cores'
    num_decimal_places = 1
    soap_owds = file_utils.find('/global/cscratch1/sd/trose/soap_run_calcs/benchmarks/vary_num_cores', 'num_cores*')
    print('soap_owds', soap_owds, flush=True)
    data_matrix = get_data(data_to_get, soap_owds, num_decimal_places=num_decimal_places, sort_by=sort_by)
    print('data_matrix', flush=True)
    print(data_matrix, flush=True)
    file_utils.write_rows_to_csv(outfile_fpath, data_matrix)

if __name__ == '__main__':
    main()