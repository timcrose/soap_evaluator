import os, sys, time, itertools, pickle
import numpy as np
import instruct
from copy import deepcopy
from python_utils import file_utils
from python_utils import math_utils
from python_utils.list_utils import sort_by_col, flatten_list
from python_utils.mpi_utils import parallel_mkdir
import quippy
from ibslib.io import read
from libmatch.environments import alchemy
from libmatch.structures import structure, structurelist
from tools import krr
import psutil
from skopt import Optimizer
from skopt.space import Real, Integer
from socket import gethostname
from mpi4py import MPI

print('imported everything', flush=True)
inst_path = sys.argv[-1]
inst = instruct.Instruct()
inst.load_instruction_from_file(inst_path)
comm_world = MPI.COMM_WORLD
MPI_ANY_SOURCE = MPI.ANY_SOURCE
num_param_combos = eval(inst.get('master', 'num_param_combos_per_replica'))
if num_param_combos is not None:
    ord_sum_of_hostname = sum([ord(c) for c in gethostname()])
    ord_sums_of_hostnames = comm_world.gather(ord_sum_of_hostname, root=0)
    if comm_world.rank == 0:
        ord_sums_of_hostnames_matrix = np.vstack((np.arange(comm_world.size), ord_sums_of_hostnames)).T
        # sorted_ord_sums are the ranks now sorted in order of physical proximity to each other (at least,
        # that's normally true when hostnames are similar).
        sorted_ord_sums = sort_by_col(ord_sums_of_hostnames_matrix, 1)[:,0]
        num_replicas = eval(inst.get('master', 'num_replicas'))
        num_ranks_per_replica = int(comm_world.size / float(num_replicas))
        if num_ranks_per_replica != (comm_world.size / float(num_replicas)):
            raise Exception('Please provide a number of replicas that evenly divides the total number of ranks.')
        color_assignment_matrix = np.vstack((np.repeat(np.arange(num_replicas), num_ranks_per_replica), sorted_ord_sums)).T
        sorted_color_assignments = sort_by_col(color_assignment_matrix, 1)[:,0]
        del color_assignment_matrix
        del sorted_ord_sums
        del ord_sums_of_hostnames_matrix
    else:
        sorted_color_assignments = None
    color = comm_world.scatter(sorted_color_assignments, root=0)
    color = int(color)
    del sorted_color_assignments
    del ord_sums_of_hostnames
    del ord_sum_of_hostname
    comm = comm_world.Split(color, comm_world.rank)
    print('comm_world.rank', comm_world.rank, 'comm.rank', comm.rank, 'color', color, 'hostname', gethostname())
    # get other root ranks list
    # This way is easy to implement but slower than using sorted_color_assignments to figure it out without communication.
    if comm_world.rank == 0:
        other_root_ranks = []
        for i in range(num_replicas - 1):
            other_root_ranks.append(comm_world.recv(source=MPI_ANY_SOURCE, tag=0))
        print('other_root_ranks', other_root_ranks, flush=True)
    elif comm.rank == 0:
        comm_world.send(comm_world.rank, dest=0, tag=0)
else:
    comm = comm_world
    other_root_ranks = []

def get_atoms_list(data_files, rank_output_path):
    '''
    data_files: list of str
        List of paths to .json files, each with a structure.
    rank_output_path: str
        path to output log file for an MPI rank

    Return: atoms_list
    atoms_list: list of quippy Atoms objects
        List of quippy Atoms objects.

    Purpose: Read each data_file in data_files into an ase atoms object and
        then convert to quippy Atoms object.
    '''
    rank_print(rank_output_path, 'num data files', len(data_files))
    rank_print(rank_output_path, 'converting jsons to quippy atoms objects')
    atoms_list = [quippy.convert.ase_to_quip(read(data_file).get_ase_atoms()) for data_file in data_files]
    rank_print(rank_output_path, 'Converted jsons to quippy atoms objects')
    return atoms_list

class SetUpParams():
    def __init__(self):
        '''
        Interprets the instruction and calls the respective attributes of inst
        '''
        inst_path = sys.argv[-1]
        inst = instruct.Instruct()
        inst.load_instruction_from_file(inst_path)
        self.inst = inst

        sname = 'master'
        self.lowmem = eval(inst.get_with_default(sname, 'lowmem', True))
        self.lowestmem = eval(inst.get_with_default(sname, 'lowestmem', True))
        self.structure_dir = inst.get(sname, 'structure_dir') # directory of jsons of structures
        self.process_list = inst.get_list(sname, 'sections')
        self.single_molecule_energy = eval(inst.get(sname, 'single_molecule_energy'))
        self.xyz_file_basename = inst.get(sname, 'xyz_file_basename')
        self.num_structures_to_use = inst.get_with_default(sname, 'num_structures_to_use', 'all')
        
        sname = 'calculate_kernel'
        self.soap_param_list = ['time', 'python']
        self.soap_param_list += [inst.get(sname, 'glosim_path')]

        glosim_soap_options = [['filename', '', None],
                                      ['separate_species', '--separate_species', False],
                                      ['exclude','--exclude', []],
                                      ['nocenter','--nocenter', []],
                                      ['envsim','--envsim', 0],
                                      ['verbose','--verbose', False],
                                      ['n','-n', 8],
                                      ['l','-l', 6],
                                      ['c','-c', 5.0],
                                      ['cotw','--cotw', 0.5],
                                      ['g','-g', 0.5],
                                      ['cw','-cw', 1.0],
                                      ['mu','--mu', 0.0],
                                      ['usekit','--usekit', False],
                                      ['gamma','--gamma', 1.0],
                                      ['zeta','--zeta', 1.0],
                                      ['kit','--kit', {}],
                                      ['alchemy_rules','--alchemy_rules', 'none'],
                                      ['kernel','--kernel', 'average'],
                                      ['peratom', '--peratom', True],
                                      ['unsoap', '--unsoap', False],
                                      ['normalize_global', '--normalize_global', False],
                                      ['onpy', '--onpy', False],
                                      ['permanenteps','--permanenteps', 0.0],
                                      ['distance','--distance', False],
                                      ['np','--np', 1],
                                      ['ij','--ij', ''],
                                      ['nlandmarks','--nlandmarks', 0],
                                      ['first','--first', 0],
                                      ['last','--last', 0],
                                      ['reffirst','--reffirst', 0],
                                      ['reflast','--reflast', 0],
                                      ['refxyz','--refxyz', ''],
                                      ['prefix','--prefix', ''],
                                      ['livek','--livek', False],
                                      ['lowmem','--lowmem', True],
                                      ['restart','--restart', True],
                                     ]

        self.soap_standalone_options = {}
        for option, option_string, default in glosim_soap_options:
            self.add_standalone_soap_params(inst, sname, option, default)
            param_to_add = self.add_to_param_list(inst, sname, option, option_string)
            if param_to_add is not None:
                self.soap_param_list += param_to_add

        sname = 'create_rect'
        self.create_rect_param_list = ['time', 'python']
        self.create_rect_param_list += [inst.get(sname, 'glosim_path')]
        for option, option_string, default in glosim_soap_options:
            param_to_add = self.add_to_param_list(inst, sname, option, option_string)
            if param_to_add is not None:
                self.create_rect_param_list += param_to_add

    def setup_krr_params(self):
        sname = 'krr'
        self.krr_param_list = ['time', 'python']
        self.krr_param_list += [self.inst.get(sname, 'krr_path')]
        self.krr_standalone_options = {}
        for option, option_string, default in [['kernels', '--kernels', ['kernel.dat']],
                                      ['props', '--props', ['en_all.dat']],
                                      ['kweights', '--kweights', ''],
                                      ['mode', '--mode', 'random'],
                                      ['ntrain', '--ntrain', 10],
                                      ['ntest', '--ntest', 10],
                                      ['ntrue', '--ntrue', 0],
                                      ['csi', '--csi', 1.0],
                                      ['sigma', '--sigma', 1e-3],
                                      ['ntests', '--ntests', 1],
                                      ['pweights', '--pweights', ''],
                                      ['refindex', '--refindex', ''],
                                      ['saveweights', '--saveweights', 'weights.dat']
                                     ]:
            self.add_standalone_krr_params(self.inst, sname, option, default)
            param_to_add = self.add_to_param_list(self.inst, sname, option, option_string)
            if param_to_add is not None:
                self.krr_param_list += param_to_add

    def setup_krr_test_params(self):
        sname = 'krr_test'
        self.krr_test_param_list = ['time', 'python']
        self.krr_test_param_list += [self.inst.get(sname, 'krr_test_path')]
        for option, option_string in [['kernels', '--kernels'],
                                      ['weights', '--weights'],
                                      ['kweights', '--kweights'],
                                      ['props', '--props'],
                                      ['csi', '--csi'],
                                      ['noidx', '--noidx']
                                     ]:

            param_to_add = self.add_to_param_list(self.inst, sname, option, option_string)
            if param_to_add is not None:
                self.krr_test_param_list += param_to_add


    def get_kernel_fname(self, sname):
        '''
        sname: str
            section name in inst.

        Note: it is assumed you are in the path to current soap directory
        Purpose: Get the filename of the kernel file. Use glob to avoid 
            many lines to derive the correct name like is done in glosim.py.
        '''
        if sname == 'krr':
            search_str = '.k'
            search_str_avoid = 'rect'
            start_path = self.kernel_calculation_path
            test_str = ''
        elif sname == 'krr_test':
            search_str = 'rect'
            search_str_avoid = 'no_strs_to_avoid'
            start_path = '.'
            test_str = 'itest' + str(self.itest) + '*'

        file_list = file_utils.glob(os.path.join(start_path, '*' + test_str))
        for fname in file_list:
            if search_str in fname and search_str_avoid not in fname:
                return fname
        return None


    def add_standalone_soap_params(self, inst, sname, option, default):
        if inst.has_option(sname, option):
            value = inst.get(sname, option)
        else:
            value = default
        try:
            value = eval(value)
        except:
            pass
        self.soap_standalone_options[option] = value


    def add_standalone_krr_params(self, inst, sname, option, default):
        if inst.has_option(sname, option):
            value = inst.get(sname, option)
        else:
            value = default
        try:
            value = eval(value)
        except:
            pass
        self.krr_standalone_options[option] = value


    def add_to_param_list(self, inst, sname, option, option_string):
        if 'kernel' in option and 'krr' in sname:
            value = self.get_kernel_fname(sname)
            if sname == 'krr' or sname == 'krr_test':
                return [option_string, value]
            return [value]
        if sname == 'krr_test':
            if 'weights' == option: 
                value = 'weights_itest' + str(self.itest) + '.dat'
                return [option_string, value]
            if 'props' == option:
                value = 'en_new_test_itest' + str(self.itest) + '.dat'
                return [option_string, value]
            
        try:
            value = inst.get(sname, option)
            if value == 'True':
                return [option_string]
            elif value == 'False':
                return None
            elif option_string == '':
                return [value]
            return [option_string, value]
        except:
            return None


def get_incomplete_tasks(kernel_memmap_path, shape):
    '''
    kernel_memmap_path: str
        file path of the memmap which is the kernel.

    shape: tuple of 2 ints
        shape of the kernel memmap.

    Return:
    incomplete_tasks: np.array, shape (number of incomplete tasks, 2) or np.array([])
        Each row of incomplete_tasks is a np.array shape (1,2) which is the index in the kerenel matrix
        of a similarity entry that has not yet been calculated.

    Notes: Because all entries of a memmap must be the same type, the default (incomplete)
        values in the kernel is -11111.0

    Purpose: Determine which, if any, similarities have not yet been computed.
    '''
    fp = np.memmap(kernel_memmap_path, dtype='float32', mode='r', shape=shape)
    incomplete_tasks_rows, incomplete_tasks_cols = np.where(fp == -11111.0)
    # zip because np.where returns as an array of row indicies and an array of col indices but we want
    # a row, col index pair (i,j)
    zipped_incomplete_tasks = list(zip(incomplete_tasks_rows, incomplete_tasks_cols))
    # sort because element i,j in the matrix is the same as j,i so we will remove the j,i
    sorted_incomplete_tasks = np.sort(zipped_incomplete_tasks)
    if sorted_incomplete_tasks.size == 0:
        return np.array([])
    # remove the j,i. Now we only have one element in incomplete tasks for every true task.
    # Note that np.unique returns the sorted unique elements.
    incomplete_tasks = np.unique(sorted_incomplete_tasks, axis=0)
    return incomplete_tasks


def modify_soap_hyperparam_list(param_list, params_to_get, params_set, dashes='-'):
    '''
    param_list: str
        List of params to pass to glosim to compute kernel.
        Ex:
sys.path.append(os.path.join(os.environ["HOME"], "python_utils"))
         'train.xyz', '--periodic', '-n', '[5, 8]', '-l', '[1]',
         '-c', '[1]', '-g', '[0.7, 1.1]', '--kernel', 'average',
         '--unsoap', '--np', '1']
    params_to_get: list
        The hyperparams: ['n','l','c','g']
    params_set: iterable
        The value for each hyperparam in params_to_get

    Return: list
        modified param string list
    Purpose: If you define lists of possible soap hyperparams, then
        you can't get the particular set until you are running
        through the for loops in soap_workflow. So, we need to 
        replace the list of values for each hyperparam by 
        the current desired value of that hyperparam.
    '''
    for i, p in enumerate(params_to_get):
        pos_p = param_list.index(dashes + p)
        param_list = param_list[: pos_p + 1] + \
              [str(params_set[i])] + \
              param_list[pos_p + 2 :]
    return param_list

def write_similarities_to_file(i, j, sij, num_structures, kernel_memmap_path):
    float32_size = 4
    offset = (i * num_structures + j) * float32_size
    fp = np.memmap(kernel_memmap_path, dtype='float32', mode='r+', offset=offset, shape=(1, 1))
    fp[:] = sij


def get_krr_task_list(param_path):
    ntrain_paths = set(file_utils.find(param_path, 'ntrain_*', recursive=True))
    # As a quick approximation, assume that if the output file "saved" exists, then this ntrain_path is done. TODO look
    # inside the files to see for sure if all of the ntests were completed to be more sure if this ntrain_path is done or not.
    saved_paths = set(file_utils.find(param_path, 'saved', recursive=True))
    krr_task_list = sorted(list(ntrain_paths - saved_paths))
    return krr_task_list


def create_krr_dirs(inst, param_path):
    '''
    inst
        Instruct object storing .conf file options
    param_path: str
        full path to hyperparameter directory. The selection methods will be in that directory.

    Return: None

    Purpose: Create selection method directories in the hyperparameter directory. Then create a tree of ntest/ntrain
        paths under each selection method according to the ntest and ntrain values provided in the .conf file. All
        permutations of ntest/ntrain will be created. Example ntest = [2,3], ntrain = [4,5] then ntest_2/ntrain_4, ntest_2/ntrain_5,
        ntest_3/ntrain_4, and ntest_3/ntrain_5 will be created. It is in these ntrain directories that krr will occur.
    '''
    ntest = eval(inst.get('krr', 'ntest'))
    ntrain = eval(inst.get('krr', 'ntrain'))
    selection_methods = eval(inst.get('krr', 'mode'))
    for selection_method in selection_methods:
        for ntest_num_structures in ntest:
            for ntrain_num_structures in ntrain:
                file_utils.mkdir_if_DNE(os.path.join(param_path, selection_method, 'ntest_' + str(ntest_num_structures), 'ntrain_' + str(ntrain_num_structures)))


def get_specific_krr_params(ntrain_path):
    # modify krr param list for this specific task. The parameters can be extracted from ntrain_path
    ntrain = int(os.path.basename(ntrain_path).split('_')[-1])
    ntest = int(ntrain_path.split('/')[-2].split('_')[-1])
    selection_method = ntrain_path.split('/')[-3]
    return ntrain, ntest, selection_method


def root_print(rank, *print_message):
    if rank == 0:
        print(print_message,flush=True)

def rank_print(rank_output_path, *print_message):
    with open(rank_output_path, mode='a') as f:
        print(print_message, file=f, flush=True)

def task_complete(i, j, num_structures, kernel_memmap_path):
    '''
    Return: bool
        True: task is already complete
        False: task is incomplete
    '''
    float32_size = 4
    offset = (i * num_structures + j) * float32_size
    fp = np.memmap(kernel_memmap_path, dtype='float32', mode='r', offset=offset, shape=(1, 1))
    print('fp[0][0]',fp[0][0], 'fp[0][0] != -11111.0', fp[0][0] != -11111.0)
    return fp[0][0] != -11111.0


def get_traversal_order(incomplete_tasks, kernel_calculation_path, rank_output_path):
    '''
    incomplete_tasks: np.array shape (num incomplete tasks, 2)
        Each row is a pair of structure indices that need their similarities evaluated.

    kernel_calculation_path: str
        Path where the kernel.dat and num_atoms_arr.npy are located

    rank_output_path: str
        path to output log file for an MPI rank

    Purpose: To save on memory, we'd like to minimize the amount of memory stored at any given time so 
        store a atomic env matrix or its repeated or tiled forms if they are needed for the present calculation
        and delete them afterwards unless needed for a future calculation in the task list. A viable algorithm is to
        order the structures "as if they were" from left to right with left being the least number of other structures with the same
        length and right being the structure with the most number of other structures with the same length. Then
        traverse the kernel matrix entries "as if they were" in row-major order. I only say "as if they were" because
        I don't want to change the actual indices of the structures.

    Return: incomplete_tasks
    incomplete_tasks: np.array shape (num incomplete tasks, 2)
        incomplete_tasks is now in the order specified above.
    '''
    # if all structures have the same number of atoms, just return incomplete_tasks unmodified
    # 1D array containing the number of atoms in each structure in data_files.
    if not isinstance(incomplete_tasks, np.ndarray):
        raise TypeError('incomplete_tasks is not of type np.ndarray but it should be. Instead it has the type', type(incomplete_tasks))
    if incomplete_tasks.shape[0] == 0:
        return incomplete_tasks
    if len(incomplete_tasks.shape) != 2:
        raise ValueError('len(incomplete_tasks.shape) != 2, but it should be. Instead it has shape', incomplete_tasks.shape)
    if incomplete_tasks.shape[1] != 2:
        raise ValueError('incomplete_tasks.shape[1] != 2, but it should be. Instead it is', incomplete_tasks.shape[1])
    num_atoms_arr = np.load(os.path.join(kernel_calculation_path, 'num_atoms_arr.npy'))
    #rank_print(rank_output_path, 'num_atoms_arr',num_atoms_arr)
    if np.all(num_atoms_arr == num_atoms_arr[0]):
        rank_print(rank_output_path, 'returning given incomplete tasks')
        return incomplete_tasks
    else:
        arr_with_idx = np.vstack((np.arange(len(num_atoms_arr)), num_atoms_arr)).T
        c = Counter(num_atoms_arr)
        argsorted_arr = np.array(sorted(arr_with_idx, key=lambda x:(c[x[1]], x[1])))[:,0]
        sorted_order_dct = {i:s for i,s in enumerate(argsorted_arr)}

    incomplete_tasks = np.array([[sorted_order_dct[i], sorted_order_dct[j]] for i,j in incomplete_tasks])
    return incomplete_tasks


def delete_some_unnecessary_matrices(loaded_soaps, matrices_to_keep, upperbound_bytes):
    '''
    loaded_soaps: dict
        Keys are structure indices with '_repeat' appended if it is repeated or '_tile' appended if it is tiled or
        neither appended if it is the raw matrix.
        Values are the matrices after having been repeated or tiled.
    matrices_to_keep: list of str
        List of names of matrices to keep for the current calculation
    upperbound_bytes: number
        An upper bound on the number of bytes needed to load the desired matrix

    Return: None
    
    Purpose:
        Update loaded_soaps by deleting matrices that are not used in the current calculation if there is not enough
        memory to load the matrices needed for the current calculation.
    '''
    # check if there is enough memory
    while dict(psutil.virtual_memory()._asdict())['available'] < upperbound_bytes:
        # delete a matrix that is stored but not used in the current calculation
        # To further improve upon what is currently being done: Could actually see if a particular matrix will be used later on and prefer to keep it if you could instead delete one that will not be used later on
        for loaded_soap in list(loaded_soaps.keys()):
            if loaded_soap not in matrices_to_keep:
                del loaded_soaps[loaded_soap]
                break


def delete_all_unnecessary_matrices(loaded_soaps, matrices_to_keep):
    '''
    loaded_soaps: dict
        Keys are structure indices with '_repeat' appended if it is repeated or '_tile' appended if it is tiled or
        neither appended if it is the raw matrix.
        Values are the matrices after having been repeated or tiled.
    matrices_to_keep: list of str
        List of names of matrices to keep for the current calculation

    Return: None

    Purpose:  Update loaded_soaps by deleting matrices that are not used in the current calculation to enable lowest possible
        memory consumption at any given time. Useful when you do not have much memory on your machine.
    '''
    for loaded_soap in list(loaded_soaps.keys()):
        if loaded_soap not in matrices_to_keep:
            del loaded_soaps[loaded_soap]
    

def load_soaps(task_idx, loaded_soaps, task_list, num_atoms_arr, global_envs_dir, lowmem, lowestmem, breathing_room_factor):
    '''
    task_idx: int
        index of current task in task_list
    loaded_soaps: dict
        Keys are structure indices with '_repeat' appended if it is repeated or '_tile' appended if it is tiled or
        neither appended if it is the raw matrix.
        Values are the matrices after having been repeated or tiled.
    task_list: np.array shape (x, 2)
        Each row is a pair of indices corresponding to structures that need their similarities evaluated.
    num_atoms_arr: np.array shape (num_structures,)
        an array of the number of atoms in each structure
    global_envs_dir: str
        path that contains the environment matrices for each structure
    lowmem: bool
        True: Check to see if you have enough memory to load a new matrix, and if not, delete stored matrices that
            are not used in the current calculation
        False: Assume that you have enough memory to load a new matrix without checking.
        Notes: Overriden by lowestmem option.
    lowestmem: bool
        True: Only load the necessary matrices to do the current calculation and delete all other stored matrices
            that are not used in the current calculation.
        False: look to lowmem to determine behavior            
        Notes: Overrides lowmem option
    breathing_room_factor: float
        amount to multiply the required available memory by in order to deem the action of loading the current matrix
        safe. Given that there are many MPI ranks trying to load matrices possibly simultaneously, this factor could 
        reasonably be as high as comm.size / node where node is the total number of nodes the program is running on
        and it is assumed that psutil just gets the available memory on the node of the requesting rank.

    Return: int, int
        num_atoms_in_i, num_atoms_in_j
        num_atoms_in_i: int
            number of atoms in the unit cell of structure i
        num_atoms_in_j: int
            number of atoms in the unit cell of structure j

    Purpose:
        Ideally: In order to keep loaded as many matrices as possible but not too many matrices so as to overflow the memory,
        this function loads only the needed matrices and deletes matrices if either they are not needed for a future calculation
        in task_list or if they are needed for a future calculation but not the current one and there is not enough memory for the current
        calculation AND those matrices used for future calculations only.

        Current: In order to keep loaded as many matrices as possible but not too many matrices so as to overflow the memory,
        this function loads only the needed matrices and deletes matrices that are not needed for the current calculation if
        there is not enough memory to load a matrix needed for the current calculation.

        If lowmem is False, load matrices needed for the current calculation but do not delete matrices that were loaded already unless lowestmem
        is True.
    '''
    """
    i: int
        Index of the structure whose matrix to load for np.repeat
    j: int
        Index of the structure whose matrix to load for np.tile
    """
    i,j = task_list[task_idx]
    if len(loaded_soaps) == 0:
        # no soaps loaded yet, go ahead and load one
        loaded_soaps[str(i)] = np.load(os.path.join(global_envs_dir, 'atomic_envs_matrix_' + str(i) + '.npy'))
    #print('loaded_soaps.keys()', loaded_soaps.keys(), flush=True)
    #print('list(loaded_soaps.keys())', list(loaded_soaps.keys()), flush=True)
    one_matrix = loaded_soaps[list(loaded_soaps.keys())[0]]
    bytes_per_atom = one_matrix.nbytes / one_matrix.shape[0]
    del one_matrix
    float32_size = 4
    num_atoms_in_i = int(num_atoms_arr[i])
    #print('num_atoms_in_i', num_atoms_in_i)
    num_atoms_in_j = int(num_atoms_arr[j])

    matrices_to_keep = [str(i), str(j), str(i) + '_repeat_' + str(num_atoms_in_j), str(j) + '_tile_' + str(num_atoms_in_i)]

    if str(i) + '_repeat_' + str(num_atoms_in_j) not in loaded_soaps:    
        if str(i) not in loaded_soaps:
            if lowestmem:
                delete_all_unnecessary_matrices(loaded_soaps, matrices_to_keep)
            elif lowmem:
                delete_some_unnecessary_matrices(loaded_soaps, matrices_to_keep, num_atoms_in_i * bytes_per_atom * breathing_room_factor)
            loaded_soaps[str(i)] = np.load(os.path.join(global_envs_dir, 'atomic_envs_matrix_' + str(i) + '.npy'))
        if lowestmem:
            delete_all_unnecessary_matrices(loaded_soaps, matrices_to_keep)
        elif lowmem:
            delete_some_unnecessary_matrices(loaded_soaps, matrices_to_keep, num_atoms_in_i * num_atoms_in_j * bytes_per_atom * breathing_room_factor)
        #print('loaded_soaps['+str(i)+']', loaded_soaps[str(i)])
        loaded_soaps[str(i) + '_repeat_' + str(num_atoms_in_j)] = np.repeat(loaded_soaps[str(i)], num_atoms_in_j, axis=0).flatten()

    #Could insert a statement here to delete loaded_soaps[str(i)] if it's not needed in any future calculations to further save on memory

    if str(j) + '_tile_' + str(num_atoms_in_i) not in loaded_soaps:
        if str(j) not in loaded_soaps:
            if lowestmem:
                delete_all_unnecessary_matrices(loaded_soaps, matrices_to_keep)
            elif lowmem:
                delete_some_unnecessary_matrices(loaded_soaps, matrices_to_keep, num_atoms_in_j * bytes_per_atom * breathing_room_factor)
            loaded_soaps[str(j)] = np.load(os.path.join(global_envs_dir, 'atomic_envs_matrix_' + str(j) + '.npy'))
        if lowestmem:
            delete_all_unnecessary_matrices(loaded_soaps, matrices_to_keep)
        elif lowmem:
            delete_some_unnecessary_matrices(loaded_soaps, matrices_to_keep, num_atoms_in_i * num_atoms_in_j * bytes_per_atom * breathing_room_factor)
        loaded_soaps[str(j) + '_tile_' + str(num_atoms_in_i)] = np.tile(loaded_soaps[str(j)], (num_atoms_in_i, 1)).flatten()

    #Could insert a statement here to delete loaded_soaps[str(j)] if it's not needed in any future calculations to further save on memory
    #Could actually see if a particular matrix will be used later on and prefer to keep it if you could instead delete one that will not be used later on
    #Could try writing and reading the repeated and tiled matrices to and from a file but this might be slower.
    return num_atoms_in_i, num_atoms_in_j


def appropriate_np_del_indices(indices, len_of_arr):
    '''
    indices: 1D iterable
        List of indices to delete in 1D array arr
    len_of_arr: int
        Length of 1D array arr

    Return: np.array, shape (num valid indices to delete from arr,)

    Purpose: If indices are passed to np.delete that are out of bounds in arr, then an error
        will be raised. To avoid this, this function determines which of the indices provided
        are valid indices to delete from arr (in bounds indices) and returns those ones only.
    '''
    return np.array(list(set([int(i) for i in indices if i in np.arange(len_of_arr) or -1 - i in np.arange(len_of_arr)])), dtype=int)


def get_weighted_avg_test_R2(avg_test_R2s):
    '''
    avg_test_R2s: np.array, shape (num selection methods,)
        Matrix where each row is the R^2 values averaged over ntests for a different selection method.
    
    Return:
        weighted_avg_test_R2: float

    Purpose: For a given hyperparameter set, I might do multiple selection methods such as iss, random, fps...
        However, I need one objective function value to tell to skopt.Optimizer. One way is to get the avg R^2
        value from each of these selection methods and create an overall score which is the weighted avg of
        these R^2 values where the weights are the relative quality of the R^2 value. For example,
        iss: R^2 = 0.90, weight = 0.90/0.90 = 1.0000
        fps: R^2 = 0.85, weight = 0.85/0.90 = 0.9444
        cur: R^2 = 0.80, weight = 0.80/0.90 = 0.8888
        overall weighted avg R^2 = (1 * 0.9 + 0.9444 * 0.85 + 0.8888 * 0.8) / (1 + 0.9444 + 0.8888) = 0.852
        This is done to not penalize too much for a particularly bad selection method result, but to still average
        to not overfit and average out some of the randomness.
    '''
    max_R2 = np.max(avg_test_R2s)
    weights = avg_test_R2s / max_R2
    weighted_avg_test_R2 = np.dot(avg_test_R2s, weights) / np.sum(weights)
    return weighted_avg_test_R2


def binding_energy(nmpc, total_energy, single_molecule_energy):
    return total_energy - (nmpc * single_molecule_energy)

def normalized_BE_by_napc(napc, nmpc, total_energy, single_molecule_energy, BE=None):
    if BE is None:
        BE = binding_energy(nmpc, total_energy, single_molecule_energy)
    return BE / float(napc)

def write_en_dat(wdir_path, structure_dir, out_fname, single_molecule_energy, num_structures_to_use, energy_name='energy'):
    '''
    wdir_path: str
        Directory to write the energy dat file to.
    structure_dir: str
        path to directory containing jsons with structures
    out_fname: str
        File name of the desired outputted dat file. (Must end with .dat)
    single_molecule_energy: float
        total energy of the single molecule used to create the structures in structure_dir (eV)
    num_structures_to_use: int or str
        number of structures to use out of all structures in structure_dir (gets the first num_structures_to_use). If
        'all', then use all structures in structure_dir
    energy_name: str
        name of the key in the structures' property dictionary containing the total energy (eV)

    Purpose: write an energy dat file which is a single column
        listing the energy of each structure.
    '''
    out_fpath = os.path.join(wdir_path, out_fname)
    num_atoms_arr_outfpath = os.path.join(wdir_path, 'num_atoms_arr.npy')
    json_files = file_utils.glob(os.path.join(structure_dir, '*.json'))
    if num_structures_to_use != 'all':
        json_files = json_files[: int(num_structures_to_use)]
    num_atoms_lst = []
    with open(out_fpath, mode='w') as f:
        for json_file in json_files:
            struct = file_utils.get_dct_from_json(json_file, load_type='load')
            if 'properties' in struct and energy_name in struct['properties']:
                en=float(struct['properties'][energy_name])
                geometry = np.array([struct['geometry'][i][:3] for i in range(len(struct['geometry']))])
                napc = len(geometry)
                num_atoms_lst.append(napc)
                nmpc = int(struct['properties']['nmpc'])
                energy = normalized_BE_by_napc(napc, nmpc, en, single_molecule_energy)
            else:
                energy = 'none'
            f.write(str(energy) + '\n')
    np.save(num_atoms_arr_outfpath, np.array(num_atoms_lst, dtype='float32'))


def soap_workflow(params):
    '''
    params: SetUpParams object
        Contains input parameters given by the user in soap.conf

    Purpose: Navigate to the various working directories that were
        already created and run through the soap workflow. This 
        consists of (1) creating the kernel, (2) using krr to 
        save weights when training, (3) creating the rectangular
        similarity matrix, (4) using krr_test to actually predict
        energies of the test set.
    '''
    inst = params.inst
    c_precision = eval(inst.get_with_default('calculate_kernel', 'c_precision', 0.5))
    g_precision = eval(inst.get_with_default('calculate_kernel', 'g_precision', 0.5))
    zeta_precision = eval(inst.get_with_default('calculate_kernel', 'zeta_precision', 0.5))
    prec_dct = {'c': c_precision, 'g': g_precision, 'zeta': zeta_precision}
    params_to_get = ['n','l','c','g','zeta']
    sname = 'krr'
    krr_results_memmap_offset = 0
    float32_size = 4
    #original working dir
    owd = os.getcwd()
    opt_pickle_dir = inst.get_with_default('master', 'opt_pickle_dir', owd)
    opt_pickle_fpath = os.path.join(opt_pickle_dir, 'opt.pickle')
    krr_results_memmap_fpath = os.path.join(owd,'krr_results.dat')
    soap_runs_dir = os.path.join(owd, 'soap_runs')
    num_param_combos = eval(inst.get('master', 'num_param_combos_per_replica'))
    
    if num_param_combos is None:
        params_list = [inst.get_list('calculate_kernel', p) for p in params_to_get]
        params_combined_iterable = itertools.product(*params_list)
        iterable = params_combined_iterable
    else:
        iterable = range(num_param_combos)
    if comm_world.rank == 0 and num_param_combos is not None:
        if os.path.exists(opt_pickle_fpath):
            with open(opt_pickle_fpath, 'rb') as f:
                opt = pickle.load(f)
        else:
            # We're using all Integer Dimension objects because it will reduce the search space, however, we want 0.5 level accuracy for now on
            # c, g, and zeta so I'm multiplying their ranges by 2 to get a value, and then later dividing that value by 2 to get the true value to 
            # pass to the kernel.
            n_lower_bound = eval(inst.get_with_default('calculate_kernel', 'n_lower_bound', 4))
            n_upper_bound = eval(inst.get_with_default('calculate_kernel', 'n_upper_bound', 14))
            l_lower_bound = eval(inst.get_with_default('calculate_kernel', 'l_lower_bound', 4))
            l_upper_bound = eval(inst.get_with_default('calculate_kernel', 'l_upper_bound', 14))
            c_lower_bound = eval(inst.get_with_default('calculate_kernel', 'c_lower_bound', 2))
            c_upper_bound = eval(inst.get_with_default('calculate_kernel', 'c_upper_bound', 40))
            g_lower_bound = eval(inst.get_with_default('calculate_kernel', 'g_lower_bound', 0.5))
            g_upper_bound = eval(inst.get_with_default('calculate_kernel', 'g_upper_bound', 40))
            zeta_lower_bound = eval(inst.get_with_default('calculate_kernel', 'zeta_lower_bound', 0.5))
            zeta_upper_bound = eval(inst.get_with_default('calculate_kernel', 'zeta_upper_bound', 7))

            print('n_lower_bound = {}, n_upper_bound = {}, l_lower_bound = {}, l_upper_bound = {}, c_lower_bound = {}, c_upper_bound = {}, g_lower_bound = {}, g_upper_bound = {}, zeta_lower_bound = {}, zeta_upper_bound = {},'.format(n_lower_bound, n_upper_bound, l_lower_bound, l_upper_bound, c_lower_bound, c_upper_bound, g_lower_bound, g_upper_bound, zeta_lower_bound, zeta_upper_bound), flush=True)
            print('c_precision = {}, g_precision = {}, zeta_precision = {}'.format(c_precision, g_precision, zeta_precision), flush=True)
            
            opt = Optimizer([
                Integer(n_lower_bound, n_upper_bound),
                Integer(l_lower_bound, l_upper_bound),
                Integer(int(c_lower_bound / c_precision), int(c_upper_bound / c_precision)),
                Integer(int(g_lower_bound / g_precision), int(g_upper_bound / g_precision)),
                Integer(int(zeta_lower_bound / zeta_precision), int(zeta_upper_bound / zeta_precision))],
                n_initial_points=50)
    
    root_print(comm_world.rank, 'using ' + str(comm.size) + ' MPI ranks on ' + str(params.num_structures_to_use) + ' structures.')
    start_time = time.time()

    for params_iter in iterable:
        start_time_param = time.time()
        param_string = ''
        if num_param_combos is None:
            params_set = params_iter
        else:
            if comm_world.rank == 0:
                params_sets = opt.ask(n_points=num_replicas, strategy='cl_max')
                root_print(comm_world.rank, 'params_sets gotten when asked:', params_sets)
                params_sets = [[math_utils.round(p,3,leave_int=True) for p in param_set] for param_set in params_sets]
                root_print(comm_world.rank, 'rounded params_sets', params_sets)
                for i, root in enumerate(other_root_ranks):
                    comm_world.send(params_sets[i + 1], dest=root, tag=1)
                params_set = params_sets[0]
            elif comm.rank == 0:
                params_set = comm_world.recv(source=0, tag=1)
            else:
                params_set = None
            params_set = comm.bcast(params_set, root=0)

        n, l, c, g, zeta = params_set
        if num_param_combos is not None:
            # Multiply by precision because we're getting 1/prec from the ask and we want prec increments out of integer types in the optimizer
            c *= c_precision
            g *= g_precision
            zeta *= zeta_precision
        params_set = [n, l, c, g, zeta]
        params_dct = {params_to_get[i]:params_set[i] for i in range(len(params_to_get))}
        for p in params_to_get:
            param_string += p
            param_to_add = str(params_dct[p])
            #c and g have floats in the name in the kernel file.
            #Format is n8-l8-c4.0-g0.3
            if p == 'g' or p == 'c' or p == 'zeta':
                param_to_add = str(float(param_to_add))
                    
            param_string += param_to_add
            if p != params_to_get[-1]:
                param_string += '-'
        
        param_path = os.path.join(soap_runs_dir, param_string)
        root_print(comm_world.rank, 'param_path', param_path)
        parallel_mkdir(comm.rank, param_path)

        rank_output_dir = os.path.join(param_path, 'output_log')
        rank_output_path = os.path.join(rank_output_dir, 'output_from_rank_' + str(comm.rank) + '.out')
        parallel_mkdir(comm.rank, rank_output_dir)

        kernel_calculation_path = os.path.abspath(os.path.join(param_path, 'calculate_kernel'))
        params.kernel_calculation_path = kernel_calculation_path
        parallel_mkdir(comm.rank, kernel_calculation_path)

        en_all_dat_fname = inst.get('krr', 'props')
        structure_dir = inst.get('master', 'structure_dir')
        num_structures_to_use = inst.get_with_default('master', 'num_structures_to_use', 'all')
        single_molecule_energy = eval(inst.get('master', 'single_molecule_energy'))
        if comm.rank == 0:
            write_en_dat(kernel_calculation_path, structure_dir, en_all_dat_fname, single_molecule_energy, num_structures_to_use)
        en_all_dat_fpath = os.path.join(kernel_calculation_path, en_all_dat_fname)
        
        # Calculate global environments if not already calculated
        global_envs_dir = os.path.join(kernel_calculation_path, 'tmpstructures')
        parallel_mkdir(comm.rank, global_envs_dir)
        comm.barrier()

        start_time_envs = time.time()
        num_structures = len(file_utils.get_lines_of_file(en_all_dat_fpath))
        
        root_print(comm_world.rank, str(num_structures) + ' structures are in the pool.')
        # make an array that will store the number of atoms in every structure.
        # Must have at least 2 ranks so that 1 can be the master rank
        ##print('num_structures', num_structures, flush=True)
        ##print('comm.size', comm.size, flush=True)
        num_tasks_per_rank = int(np.ceil(float(num_structures) / float(comm.size)))
        
        if os.path.exists(global_envs_dir):
            global_envs_incomplete_tasks = np.array(sorted(list(set(np.arange(num_structures)) - set([int(file_utils.fname_from_fpath(fpath).split('_')[1]) for fpath in file_utils.find(global_envs_dir, '*.dat')]))))
        else:
            global_envs_incomplete_tasks = np.arange(num_structures)
        if comm.rank < comm.size - 1:
            global_envs_incomplete_tasks = global_envs_incomplete_tasks[comm.rank * num_tasks_per_rank : (comm.rank + 1) * num_tasks_per_rank]
        else:
            global_envs_incomplete_tasks = global_envs_incomplete_tasks[comm.rank * num_tasks_per_rank :]
        rank_print(rank_output_path, 'my assigned global environments to calculate:', global_envs_incomplete_tasks)

        alchem = alchemy(mu=params.soap_standalone_options['mu'])
        os.chdir(kernel_calculation_path)
        data_files = file_utils.glob(os.path.join(params.structure_dir, '*.json'))
        if params.num_structures_to_use != 'all':
            data_files = data_files[: int(params.num_structures_to_use)]
        
        # global_envs_incomplete_tasks is a np.array of tasks where each task is the index in the structure list of a structure that still
        # needs its global soap descriptor calculated.
        # Calculate the global soap descriptor for each structure whose index is in global_envs_incomplete_tasks
        if len(global_envs_incomplete_tasks) > 0:
            start = global_envs_incomplete_tasks[0]
            rank_print(rank_output_path, 'Beginning computation of global envs with start =', start)
            if global_envs_incomplete_tasks[-1] == start:
                stop = start + 1
            else:
                stop = global_envs_incomplete_tasks[-1] + 1
            rank_print(rank_output_path, 'About to get atoms list')
            al = get_atoms_list(data_files[start : stop], rank_output_path)
            rank_print(rank_output_path, 'Got atoms list')
            sl = structurelist()
            sl.count = start # Set to the global starting index
            just_env_start_time = time.time()
            for at in al:
                rank_print(rank_output_path, 'computing atomic env matrix for structure indexed by', sl.count)
                rank_print(rank_output_path, 'Using parameters', 'c = {}, cotw = {}, n = {}, l = {}, g = {}, cw = {}, zeta = {}, nocenter = {}, exclude = {}, unsoap = {}'.format(c, params.soap_standalone_options['cotw'], n, l, g, params.soap_standalone_options['cw'], zeta, params.soap_standalone_options['nocenter'], params.soap_standalone_options['exclude'], params.soap_standalone_options['unsoap']))
                si = structure(alchem)
                si.parse(at, c, params.soap_standalone_options['cotw'], n, l, g, params.soap_standalone_options['cw'], params.soap_standalone_options['nocenter'], params.soap_standalone_options['exclude'], unsoap=params.soap_standalone_options['unsoap'], kit=params.soap_standalone_options['kit'])
                sl.append(si.atomic_envs_matrix)
            rank_print(rank_output_path, 'time to compute envs for {} structures: {}'.format(stop - start, time.time() - just_env_start_time))
        start_time_kernel = time.time()
        rank_print(rank_output_path,'time to read input structures and calculate environments', start_time_kernel - start_time_envs)
        root_print(comm_world.rank,'time to read input structures and calculate environments', start_time_kernel - start_time_envs)
        
        # Calculate kernel if not already calculated
        kernel_memmap_path = os.path.join(kernel_calculation_path, 'kernel.dat')
        if comm.rank == 0:
            if os.path.exists(kernel_memmap_path):
                incomplete_tasks = get_incomplete_tasks(kernel_memmap_path, (num_structures, num_structures))
                print('got incomplete_tasks from get_incomplete_tasks', flush=True)
            else:
                incomplete_tasks_rows, incomplete_tasks_cols = np.triu_indices(num_structures)
                print('incomplete_tasks_rows.shape', incomplete_tasks_rows.shape, 'incomplete_tasks_cols', incomplete_tasks_cols, flush=True)
                incomplete_tasks = np.array(list(zip(incomplete_tasks_rows, incomplete_tasks_cols)))
                print('got incomplete_tasks from zip', flush=True)
                fp = np.memmap(kernel_memmap_path, dtype='float32', mode='w+', shape=(num_structures, num_structures))
                # Initialize to -11111.0 just so we know that if it's -11111.0 that it's incomplete
                fp[:] = np.ones((num_structures, num_structures), dtype='float32') * -11111.0
                del fp
            print('getting kernel traversal order', flush=True)
            print('b4 incomplete_tasks.shape', incomplete_tasks.shape, flush=True)
            print('b4 incomplete_tasks.size', incomplete_tasks.size, flush=True)
        comm.barrier()
        if comm.rank != 0:
            incomplete_tasks = get_incomplete_tasks(kernel_memmap_path, (num_structures, num_structures))

        incomplete_tasks = get_traversal_order(incomplete_tasks, kernel_calculation_path, rank_output_path)
        num_tasks_per_rank = int(np.ceil(float(len(incomplete_tasks)) / float(comm.size)))
        if comm.rank < comm.size - 1:
            incomplete_tasks = incomplete_tasks[comm.rank * num_tasks_per_rank : (comm.rank + 1) * num_tasks_per_rank]
        else:
            incomplete_tasks = incomplete_tasks[comm.rank * num_tasks_per_rank :]
        rank_print(rank_output_path, 'For kernel computation, got incomplete_tasks', incomplete_tasks)
        loaded_soaps = {}
        num_atoms_arr = np.load(os.path.join(kernel_calculation_path, 'num_atoms_arr.npy'))
        # incomplete_tasks should be traversal order of the kernel matrix tasks such that it should be a np.array with shape (num_tasks_per_rank, 2) where
        # each row has (as the first element) the struct index of a structure which should get its atomic env matrix repeated and (as the second element)
        # the struct index of a structure which should get its atomic env matrix tiled.
        # incomplete_tasks is np.array with shape usually being (num_tasks_per_rank, 2) where each row is a pair of structures that need their similarities evaluated.
        breathing_room_factor = min([12.0, comm.size])
        just_kernel_start_time = time.time()
        for task_idx in range(len(incomplete_tasks)):
            num_atoms_in_i, num_atoms_in_j = load_soaps(task_idx, loaded_soaps, incomplete_tasks, num_atoms_arr, global_envs_dir, params.lowmem, params.lowestmem, breathing_room_factor)
            i,j = incomplete_tasks[task_idx]
            sij = np.dot(loaded_soaps[str(i) + '_repeat_' + str(num_atoms_in_j)], loaded_soaps[str(j) + '_tile_' + str(num_atoms_in_i)]) ** zeta
            write_similarities_to_file(i, j, sij, num_structures, kernel_memmap_path)
            write_similarities_to_file(j, i, sij, num_structures, kernel_memmap_path)
        start_time_krr = time.time()
        rank_print(rank_output_path,'time to calculate kernel', start_time_krr - start_time_kernel)
        root_print(comm_world.rank,'time to calculate kernel', start_time_krr - start_time_kernel)
        rank_print(rank_output_path, 'time to compute kernel entries for {} entries: {}'.format(len(incomplete_tasks), start_time_krr - just_kernel_start_time))
        
        # krr. Currently haven't implemented krr_test into the mpi version.
        if 'krr' not in params.process_list:
            rank_print(rank_output_path, 'no krr; returning')
            return
        # barrier here to ensure kernel is completed before beginning krr.
        if comm.rank == 0:
            #create krr working directories (ntrain path)
            create_krr_dirs(inst, param_path)
        comm.barrier()
        rank_print(rank_output_path, 'entering krr')

        krr_task_list = get_krr_task_list(param_path)
        num_tasks_per_rank = int(np.ceil(float(len(krr_task_list)) / float(comm.size)))
        rank_print(rank_output_path, 'overall krr_task_list', krr_task_list)
        if comm.rank < comm.size - 1:
            krr_task_list = krr_task_list[comm.rank * num_tasks_per_rank : (comm.rank + 1) * num_tasks_per_rank]
        else:
            krr_task_list = krr_task_list[comm.rank * num_tasks_per_rank :]
        params.setup_krr_params()
        outfile_path = inst.get(sname, 'outfile')
        props_fname = params.krr_param_list[params.krr_param_list.index('--props') + 1]
        props_path = os.path.join(kernel_calculation_path, os.path.basename(os.path.abspath(props_fname)))
        #sys_stdout_original = sys.stdout
        # krr_task_list is np.array with shape (,number of tasks) where each element is an ntrain path.
        rank_print(rank_output_path, 'krr_task_list recieved in krr', krr_task_list)
        just_krr_start_time = time.time()
        my_krr_results = []
        for ntrain_path in krr_task_list:
            rank_print(rank_output_path, 'doing krr for ntrain_path', ntrain_path)
            one_krr_start_time = time.time()
            os.chdir(ntrain_path)
            ntrain, ntest, selection_method = get_specific_krr_params(ntrain_path)
            # do krr
            #sys.stdout = open(outfile_path, 'w')
            avg_test_R2 = krr.main(kernels=[kernel_memmap_path], props=[props_path], kweights=params.krr_standalone_options['kweights'], mode=selection_method, ntrain=ntrain,
                            ntest=ntest, ntrue=params.krr_standalone_options['ntrue'], csi=params.krr_standalone_options['csi'],
                            sigma=params.krr_standalone_options['sigma'], ntests=params.krr_standalone_options['ntests'], 
                            savevector=params.krr_standalone_options['saveweights'], refindex=params.krr_standalone_options['refindex'],
                            inweights=params.krr_standalone_options['pweights'])
            rank_print(rank_output_path, 'Completed krr for above ntrain_path')
            my_krr_results.append(params_set + [avg_test_R2])
            rank_print(rank_output_path, 'time to compute one krr with selection method {} is: {}'.format(selection_method, time.time() - one_krr_start_time))
        end_time_krr = time.time()
        #sys.stdout.close()
        #sys.stdout = sys_stdout_original
        rank_print(rank_output_path,'time for krr', end_time_krr - start_time_krr)
        rank_print(rank_output_path,'time for param', end_time_krr - start_time_param)
        root_print(comm_world.rank,'time for krr', end_time_krr - start_time_krr)
        root_print(comm_world.rank,'time for param', end_time_krr - start_time_param)
        rank_print(rank_output_path, 'time to compute krr {} times: {}'.format(len(krr_task_list), end_time_krr - just_krr_start_time))
        rank_print(rank_output_path, 'my_krr_results', my_krr_results)
        # Get test set weighted R^2. This is the R^2 gotten by each sampling method averaged over ntests and then weighted by
        # the fraction of the max of these R^2 values among all sampling methods tried. This will be the objective function
        # value given to skopt.Optimizer to optimize hyperparameters. The average R^2 gotten by each sampling method will only
        # be recorded for the last ntest/ntrain combo so it's suggested to just have one ntrain and one ntest value when doing
        # hyperparameter optimization.
        krr_results = comm.gather(my_krr_results, root=0)
        if comm.rank == 0:
            krr_results_matrix = np.array(flatten_list(krr_results))
            weighted_avg_test_R2 = get_weighted_avg_test_R2(krr_results_matrix[:,-1])
            krr_results_list = list(krr_results_matrix[0][:-1]) + [weighted_avg_test_R2]
            rank_print(rank_output_path, 'krr_results_list', krr_results_list)

        if comm_world.rank == 0:
            krr_results_matrices = [krr_results_list]
            for root in other_root_ranks:
                krr_results_matrices.append(comm_world.recv(source=root, tag=2))
            overall_krr_results_matrix = np.vstack(krr_results_matrices)
            root_print(comm_world.rank, 'overall_krr_results_matrix', overall_krr_results_matrix)
            # append this matrix to the memmap which has all such matrices saved (optional)
            if os.path.exists(krr_results_memmap_fpath):
                mode = 'r+'
            else:
                mode = 'w+'
            krr_results_memmap = np.memmap(krr_results_memmap_fpath, dtype='float32', mode=mode, offset=krr_results_memmap_offset, shape=overall_krr_results_matrix.shape)
            krr_results_memmap[:] = overall_krr_results_matrix
            del krr_results_memmap
            krr_results_memmap_offset += overall_krr_results_matrix.size * float32_size
            root_print(comm_world.rank, 'overall_krr_results_matrix', overall_krr_results_matrix)
            if num_param_combos is not None:
                # Divide by precision because we're getting 1/prec from the ask and we want prec increments out of integer types in the optimizer
                overall_krr_results_matrix[:,2] = overall_krr_results_matrix[:,2] / c_precision
                overall_krr_results_matrix[:,3] = overall_krr_results_matrix[:,3] / g_precision
                overall_krr_results_matrix[:,4] = overall_krr_results_matrix[:,4] / zeta_precision
                root_print(comm_world.rank, 'overall_krr_results_matrix after prec', overall_krr_results_matrix)
                root_print(comm_world.rank, 'list(map(list, np.int64(np.around(overall_krr_results_matrix[:,:-1]))))', np.int64(np.around(list(map(list, overall_krr_results_matrix[:,:-1])))))
                # Use np.int64 since we're using all Integer Dimension objects
                opt.tell(list(map(list, overall_krr_results_matrix[:,:-1])), list(overall_krr_results_matrix[:,-1]))
                with open(opt_pickle_fpath, 'wb') as f:
                    pickle.dump(opt, f)
        elif comm.rank == 0:
            comm_world.send(krr_results_list, dest=0, tag=2)


    end_time = time.time()
    rank_print(rank_output_path,'total time', end_time - start_time)
    root_print(comm_world.rank,'total time', end_time - start_time)
                        


if __name__ == '__main__':
    params = SetUpParams()
    #print(params.soap_param_list)
    soap_workflow(params)
    #print(params.krr_param_list)
