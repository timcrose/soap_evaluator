import instruct, os, shutil, sys, random, copy, itertools
from python_utils import file_utils


def write_en_dat(wdir_path, xyz_fpath, out_fname):
    '''
    wdir_path: str
        Directory to write the energy dat file to.

    xyz_fpath: str
        Full file path to the source .xyz file.

    out_fname: str
        File name of the desired outputted dat file. (Must end with .dat)

    Purpose: write an energy dat file which is a single column
        listing the energy of each structure.
    '''
    out_fpath = os.path.join(wdir_path, out_fname)
    os.system("grep energy " + xyz_fpath + " | awk '{print $10}' | sed s/energy=//g > " + out_fpath)


def create_new_working_dirs(inst):
    '''
    inst: contains conf file params

    Purpose: Delete any current folders under each dir with old runs
        of the same parameter sets and create new ones (if desired). The structure
        of the folders are designed for calculating pairwise kernels:
        Root: pairwise_kernels, depth=1: param_path, depth=2:
        species (e.g. target1).
    '''
    owd = os.getcwd()
    soap_runs_dir = os.path.join(owd, 'pairwise_kernels')
    sname = 'krr'

    params_to_get = ['n','l','c','g']
    params_list = [inst.get_list('calculate_kernel', p) for p in params_to_get]
    params_combined_iterable = itertools.product(*params_list)

    for params_set in params_combined_iterable:
        param_string = ''
        for i, p in enumerate(params_to_get):
            param_string += p
            param_to_add = str(params_set[i])
            #c and g have floats in the name in the kernel file.
            #Format is n8-l8-c4.0-g0.3
            if p == 'g' or p == 'c':
                param_to_add = str(float(param_to_add))
            param_string += param_to_add
            if p != params_to_get[-1]:
                param_string += '-'

        param_path = os.path.join(soap_runs_dir, param_string)
        if os.path.isdir(param_path):
            answer = raw_input('Really delete all contents of ' + param_path 
                            + '? (y/n)')

            if answer == 'y':
                file_utils.rm(param_path)
        file_utils.mkdir_if_DNE(param_path)

        species = inst.get('calculate_kernel', 'species')
        calculate_kernel_path = os.path.join(param_path, species)
        file_utils.mkdir_if_DNE(calculate_kernel_path)

    
def read_dataset(dataset_fname):
    '''
    dataset_fname: str
        filename of the xyz datafile containing all structures.

    return: list
        each element of lines is a line in dataset_fname
    '''
    with open(dataset_fname) as f:
        lines = f.readlines()
    return lines

def get_dataset_size(dataset):
    '''
    dataset: list
        List of lines of an xyz file

    return: list, int.
        A list of line numbers (0-based index) of each structure, Number of structures in the xyz file
    
    Notes: The number of atoms in the first sructure must be line 0 and no newlines aside from the
       required xyz file ones.
    '''

    num_lines = len(dataset)

    line_num = 0
    structs_idx = []
    while line_num < num_lines:
        line = dataset[line_num].split()
        num_atoms = int(line[0])
        lines_per_struct = num_atoms + 2
        structs_idx.append(line_num)
        line_num += lines_per_struct

    return structs_idx, len(structs_idx), lines_per_struct

def write_separate_test_and_train_xyz_files(test_idx_lines, train_idx_lines, wdir_path):
    '''
    test_idx_lines: list
        list of lines to write to test.xyz
    train_idx_lines: list
        list of lines to write to train.xyz
    wdir_path: str
        path to directory to write the xyz files to.

    Purpose: write test.xyz and train.xyz to wdir_path
    '''
    with open(os.path.join(wdir_path, 'test.xyz'), 'w') as f:
        f.writelines(test_idx_lines)
    with open(os.path.join(wdir_path, 'train.xyz'), 'w') as f:
        f.writelines(train_idx_lines)

def write_separate_test_and_train_en_dats(wdir_path):
    '''
    wdir_path: str
        path to directory to write the energy dat files to. This
        is the same directory as contains the xyz files.

    Purpose: write energy dat files which are a single column
        listing the energy of each structure.
    '''
    os.chdir(os.path.abspath(wdir_path))
    os.system("grep energy test.xyz | awk '{print $10}' | sed s/energy=//g > en_test.dat")
    os.system("grep energy train.xyz | awk '{print $10}' | sed s/energy=//g > en_train.dat")
    #os.system("grep energy test.xyz | awk '{print $11}' | sed s/nmpc=//g > nmpc_test.dat")
    #os.system("grep energy train.xyz | awk '{print $11}' | sed s/nmpc=//g > nmpc_train.dat")


def create_train_test_splits(inst, owd):
    '''
    inst: contains conf file params

    Purpose: Create train.xyz and test.xyz (which contain a collection of structures) from
        the larger xyz file with all structures (all_xyz_structs_fname) according to
        test_num_structs and train_num_structs. Note, train_num_structs is percentage remaining after taking away
        the test structures. Place train.xyz and test.xyz in the already-created 
        working directories (created by create_new_working_dirs()).
    '''
    sname = 'cross_val'

    dataset_fname = inst.get(sname, 'all_xyz_structs_fname')

    dataset = read_dataset(dataset_fname)

    structs_idx, total_dataset_size, lines_per_struct = get_dataset_size(dataset)

    params_to_get = ['n','l','c','g']
    params_list = [inst.get_list('train_kernel', p) for p in params_to_get]
    params_combined_iterable = itertools.product(*params_list)

    for selection_method in inst.get_list(sname, 'selection_methods'):
        selection_method_path = os.path.join(owd, selection_method)

        for params_set in params_combined_iterable:
            param_string = ''
            for i, p in enumerate(params_to_get):
                param_string += p
                param_to_add = str(params_set[i])
                #c and g have floats in the name in the kernel file.
                #Format is n8-l8-c4.0-g0.3
                if p == 'g' or p == 'c':
                    param_to_add = str(float(param_to_add))
                param_string += param_to_add
                if p != params_to_get[-1]:
                    param_string += '-'

            param_path = os.path.join(selection_method_path, param_string)

            for test_num_structs in inst.get_list(sname, 'test_num_structs'):
                test_num_structs_path = os.path.join(param_path, 'test_num_structs_' + str(test_num_structs))

                test_size = int(test_num_structs)

                for train_num_structs in inst.get_list(sname, 'train_num_structs'):
                    train_num_structs_path = os.path.join(test_num_structs_path, 'train_num_structs_' + str(train_num_structs))

                    train_size = int(train_num_structs)
                    if train_size > total_dataset_size - test_size:
                        raise ValueError('train_size must be <= total_dataset_size - test_size. train_size: ' +
                                       str(train_size) + ' test_size: ' + str(test_size) + ' total_dataset_size: ' + 
                                       str(total_dataset_size))


                    for test_num in range(int(inst.get(sname, 'test_num'))):
                        test_num_path = os.path.join(train_num_structs_path, 'test_num_' + str(test_num))

                        if selection_method == 'random':
                            test_idx_list = random.sample(range(total_dataset_size), test_size)
                        test_idx_lines = []
                        for test_idx in test_idx_list:
                            test_idx_lines += dataset[test_idx * lines_per_struct : (test_idx + 1) * lines_per_struct]

                        for train_num in range(int(inst.get(sname, 'train_num'))):
                            train_num_path = os.path.join(test_num_path, 'train_num_' + str(train_num))
                        
                            if selection_method == 'random':
                                train_idx_list = []
                                while len(train_idx_list) != train_size:
                                    random_idx = random.randint(0, len(structs_idx) - 1)
                                    if random_idx not in test_idx_list:
                                        train_idx_list.append(random_idx)
                            train_idx_lines = []
                            for train_idx in train_idx_list:
                                train_idx_lines += dataset[train_idx * lines_per_struct : (train_idx + 1) * lines_per_struct]

                            write_separate_test_and_train_xyz_files(test_idx_lines, train_idx_lines, train_num_path)
                            write_separate_test_and_train_en_dats(train_num_path)
def main():
    '''
    Interprets the instruction and calls the respective attributes of inst
    '''
    owd = os.getcwd()

    inst_path = sys.argv[-1]
    inst = instruct.Instruct()
    inst.load_instruction_from_file(inst_path)

    create_new_working_dirs(inst)
    #create_train_test_splits(inst, owd)

if __name__ == '__main__':
    main()
