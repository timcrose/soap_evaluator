import os, datetime, time, glob, sys
import instruct
sys.path.append(os.path.join(os.environ["HOME"], "python_utils")))
import file_utils2 as file_utils

def get_num_structs_in_trainxyz_file(path, napc):
    '''
    path: str
        path to a file where that file's directory also contains train.xyz
    napc: int
        number of atoms per unit cell. Currently the program only
        supports training sets with the same napc for each
        structure.
    '''
    train_xyz_fpath = path[:path.rfind('/')] + '/train.xyz'
    with open(train_xyz_fpath) as f:
        lines = f.readlines()
    num_structs = float(len(lines)) / float(napc + 2)
    if int(num_structs) != num_structs:
        raise IOError('check napc, each struct must have the same napc. napc = ' + str(napc))
    return int(num_structs)

def get_num_seconds_from_time_str(time_str):
    #decimal place number of seconds mess up formatting so add it separately.
    if '.' in time_str:
        fractional_sec = float(time_str[time_str.rfind('.'):])
        time_str = time_str.split('.')[0]
    else:
        fractional_sec = 0.0
    try:
        x = time.strptime(time_str,'%H:%M:%S')
    except:
        x = time.strptime(time_str,'%M:%S')
    #Gives the number of seconds (type=float) represented by time_str
    secs = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    return secs + fractional_sec

def get_elapsed_time(fpath):
    with open(fpath) as f:
        lines = f.readlines()
    secs = None
    for line in lines:
        if 'elapsed' in line:
            split_line = line.split()
            time_str = split_line[2][:split_line[2].find('elapsed')]
            secs = get_num_seconds_from_time_str(time_str)
    if secs is None:
        raise ValueError('secs is None. fpath: ' + fpath)
    return secs

def average_data_dct(data_dct):
    for key in data_dct:
        data_dct[key] = round(sum(data_dct[key]) / float(len(data_dct[key])), 2)
    return data_dct

def main():
    search_str = 'soaps'
    search_fname = 'soaps.out'
    napm = 13
    nmpc = 2
    napc = napm * nmpc

    inst_path = sys.argv[-1]
    inst = instruct.Instruct()
    inst.load_instruction_from_file(inst_path)
    
    owd = os.getcwd()
    sname = 'cross_val'
    for selection_method in inst.get_list(sname, 'selection_methods'):
        selection_method_path = os.path.join(owd, selection_method)
        param_strings = [os.path.basename(param_path) for param_path in file_utils.glob(os.path.join(selection_method_path, '*'))]
        for param_string in param_strings:
            param_path = os.path.join(selection_method_path, param_string)

            test_num_structs_paths = file_utils.glob(os.path.join(param_path, '*'))
            for test_num_structs_path in test_num_structs_paths:
                data_dct = {}
                test_num_structs = int(os.path.basename(test_num_structs_path).split('_')[-1])
                train_num_structs_paths = file_utils.glob(os.path.join(test_num_structs_path, '*'))
                for train_num_structs_path in train_num_structs_paths:
                    train_num_structs = int(os.path.basename(train_num_structs_path).split('_')[-1])

                    test_num_paths = file_utils.glob(os.path.join(train_num_structs_path, '*'))
                    for test_num_path in test_num_paths:
                        train_num_paths = file_utils.glob(os.path.join(test_num_path, '*'))
                        for train_num_path in train_num_paths:
                            search_fpath = os.path.join(train_num_path, search_fname)
                            num_structs = get_num_structs_in_trainxyz_file(search_fpath, napc)
                            secs = get_elapsed_time(search_fpath)
                            if num_structs not in data_dct:
                                data_dct[num_structs] = [secs]
                            else:
                                data_dct[num_structs].append(secs)

                avg_data_dct = average_data_dct(data_dct)
                avg_data_list = [[key, avg_data_dct[key]] for key in avg_data_dct]
                avg_data_list = sorted(avg_data_list, key=lambda x:x[0])
                file_utils.write_rows_to_csv('average_kernel_' + search_str + '_timings_' + selection_method + '_' + param_string + '_test_num_structs_' + str(test_num_structs) + '.csv', avg_data_list)


if __name__ == '__main__':
    main()
