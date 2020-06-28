from warnings import filterwarnings
filterwarnings('ignore', message='The objective has been evaluated at this point before.')
filterwarnings('ignore', category=FutureWarning)
filterwarnings('ignore', category=DeprecationWarning)
import os, sys, time, itertools, psutil
import numpy as np
import instruct
from copy import deepcopy
from collections import Counter
from python_utils import file_utils, math_utils, list_utils, mpi_utils
from put_struct_dir_pct_in_arr import put_struct_dir_pct_in_arr
from skopt_multiple_ask_before_tell import get_search_space_size, ask_opt, tell_opt
from skopt import Optimizer
from skopt.space import Real, Integer
import quippy
from ibslib.io import read
from glosim_trose_modified.libmatch.environments import alchemy
from glosim_trose_modified.libmatch.structures import structure, structurelist
from glosim_trose_modified.tools import krr
from socket import gethostname
from mpi4py import MPI

#print('imported everything', flush=True)
inst_path = sys.argv[-1]
inst = instruct.Instruct()
inst.load_instruction_from_file(inst_path)
comm_world = MPI.COMM_WORLD
MPI_ANY_SOURCE = MPI.ANY_SOURCE
num_param_combos = eval(inst.get('master', 'num_param_combos_per_replica'))
ranks_per_node = eval(inst.get_with_default('master', 'ranks_per_node', 68))
if comm_world.size % ranks_per_node != 0 and comm_world.size > ranks_per_node:
    raise ValueError('Total number of ranks must evenly divisible by the number of ranks per node.'
                    + 'ranks_per_node currently set to', ranks_per_node, ', to change it, set'     
                    + 'ranks_per_node in the soap.conf file under the "master" section.')

num_nodes = comm_world.size / ranks_per_node
num_replicas = eval(inst.get_with_default('master', 'num_replicas', 1))
num_ranks_per_replica = int(comm_world.size / float(num_replicas))
if 0 != num_nodes % num_replicas:
    raise Exception('Please provide a number of replicas that evenly divides the total number of nodes. Also, check that you set ranks_per_node correctly. num_nodes:', num_nodes, 'num_replicas:',
num_replicas, 'comm_world.size', comm_world.size, 'ranks_per_node', ranks_per_node)

breathing_room_factor_per_node = eval(inst.get_with_default('master', 'breathing_room_factor_per_node', 1.2))
if num_param_combos is not None:
    ord_sum_of_hostname = sum([ord(c) for c in gethostname()])
    ord_sums_of_hostnames = comm_world.gather(ord_sum_of_hostname, root=0)
    if comm_world.rank == 0:
        ord_sums_of_hostnames_matrix = np.vstack((np.arange(comm_world.size), ord_sums_of_hostnames)).T
        # sorted_ord_sums are the ranks now sorted in order of physical proximity to each other (at least,
        # that's normally true when hostnames are similar).
        sorted_ord_sums = list_utils.sort_by_col(ord_sums_of_hostnames_matrix, 1)[:,0]
        color_assignment_matrix = np.vstack((np.repeat(np.arange(num_replicas), num_ranks_per_replica), sorted_ord_sums)).T
        sorted_color_assignments = list_utils.sort_by_col(color_assignment_matrix, 1)[:,0]
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
# rank 0's hostname will be the 0-th index of rank_hostnames etc
rank_hostnames = comm.gather(gethostname(), root=0)
if comm.rank == 0:
    rank_hostnames = np.array(rank_hostnames)

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
        self.structure_dirs = inst.get_list(sname, 'structure_dirs') # directories of jsons of structures
        self.process_list = inst.get_list(sname, 'sections')
        self.single_molecule_energy = eval(inst.get(sname, 'single_molecule_energy'))
        self.num_structures_to_use = inst.get_with_default(sname, 'num_structures_to_use', 'all')
        self.verbose = eval(inst.get_with_default(sname, 'verbose', 'True'))
        self.max_mem = inst.get_with_default(sname, 'max_mem', 'all')
        self.node_mem = float(inst.get_with_default(sname, 'node_mem', 128.0))
        
        if self.num_structures_to_use != 'all':
            self.num_structures_to_use = int(self.num_structures_to_use)

        if self.max_mem != 'all':
            self.max_mem = float(self.max_mem)
        
        sname = 'calculate_kernel'
        self.user_num_atoms_arr = eval(inst.get_with_default(sname, 'user_num_atoms_arr', 'None'))
        self.lowmem = eval(inst.get_with_default(sname, 'lowmem', True))
        self.lowestmem = eval(inst.get_with_default(sname, 'lowestmem', True))
        self.soap_param_list = ['time', 'python']
        self.soap_param_list += [inst.get_with_default(sname, 'glosim_path', 'no_glosim_path_provided')]

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

        sname = 'krr_test'
        if sname in self.process_list:
            self.num_test_structs_list = self.inst.get_list(sname, 'num_test_structs_list')
            self.test_structs_dirs = self.inst.get_list(sname, 'test_structs_dirs')
            self.test_prop_fname = inst.get_with_default(sname, 'test_prop_fname', 'predicted_properties.npy')

    
    def setup_krr_test_params(self):
        sname = 'krr_test'
        self.krr_test_standalone_options = {}
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
                                      ['saveweights', '--saveweights', 'weights.npy']
                                     ]:
            self.add_standalone_krr_test_params(self.inst, sname, option, default)


    def setup_krr_params(self):
        sname = 'krr'
        if not self.inst.has_section(sname):
            return
        self.test_struct_dir_pct_lst = self.inst.get_list(sname, 'test_struct_dir_pct_lst')
        self.train_struct_dir_pct_lst = self.inst.get_list(sname, 'train_struct_dir_pct_lst')
        self.selection_methods = list(set(eval(inst.get('krr', 'mode'))))
        self.krr_param_list = ['time', 'python']
        self.krr_param_list += [self.inst.get_with_default(sname, 'krr_path', 'no krr_path_provided')]
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


    def add_standalone_krr_test_params(self, inst, sname, option, default):
        if inst.has_option(sname, option):
            value = inst.get(sname, option)
        else:
            value = default
        try:
            value = eval(value)
        except:
            pass
        self.krr_test_standalone_options[option] = value


    def add_to_param_list(self, inst, sname, option, option_string):
        if 'kernel' in option and 'krr' in sname:
            value = 'kernel.dat'
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


def write_similarities_to_rect_file(num_structs_in_kernel, num_test_structs, similarities, rect_kernel_memmap_path, rank_output_path=None):
    '''
    num_structs_in_kernel: int
        This is the number of structures in the kernel used to train krr. num_structs_in_kernel is used to convert given
        rect_kernel_indices to the one described below.

    num_test_structs: int
        This is the number of structures you are predicting property values for.

    similarities: list or 1D array of float
        Must have same length as kernel_indices. List of kernel values when the similarity is measured
        between structure i and structure j. The ordering of of similarities must be consistent such that
        the rect matrix = similarities.reshape((num_test_structs, num_structs_in_kernel))

    rect_kernel_memmap_path: str
        Path of the kernel np.memmap file which stores the kernel values in its matrix.

    rank_output_path: str or None
        Path to the log file for a single MPI rank. If None, no messages will be printed.
    
    Return: None

    Purpose: Write all similarities to the rect_kernel_memmap_path file.
        This is much more efficient than writing similarities one at a time. The np.memmap is read
        all at once and the values are assigned all at once. Because no other MPI rank will have the same
        indices, no clashing should occur.
    '''

    if rank_output_path is not None:
        start_time = time.time()

    if len(similarities) == 0:
        return

    float32_size = 4
    fp = np.memmap(rect_kernel_memmap_path, dtype='float32', mode='r+', shape=(num_test_structs, num_structs_in_kernel))
    if rank_output_path is not None:
        rank_print(rank_output_path, 'time to load rect_kernel_memmap_path', time.time() - start_time)
        start_time = time.time()
    
    fp[:] = np.array(similarities).reshape((num_test_structs, num_structs_in_kernel))
    if rank_output_path is not None:
        rank_print(rank_output_path, 'time to write to rect_kernel_memmap_path', time.time() - start_time)


def write_similarities_to_file(kernel_indices, similarities, kernel_memmap_path, num_structures=None, rank_output_path=None):
    '''
    kernel_indices: np.array, shape (number of pairs of structures that you have similarities for, 2)
        Each row is an i,j pair such that the the i,j th element of the kernel matrix should be 
        populated with the corresponding value in the input similarities. i.e. the kth row corresponds to
        the value the i,j th position in the kernel that similarities[k] will supply. You should be
        able to index the kernel by using the entire kernel_indices matrix. Therefore, kernel_indices 
        must have the same length as similarities.

    similarities: list or 1D array of float
        Must have same length as kernel_indices. List of kernel values when the similarity is measured
        between structure i and structure j. The order of similarities must therefore be the same order
        as the i,j pairs in kernel_indices.

    kernel_memmap_path: str
        Path of the kernel np.memmap file which stores the kernel values in its matrix.

    num_structures: int or None
        If int, it is the number of structures in the kernel such that the shape of the kernel is
        (num_structures,num_structures).
        If None, determine the shape by reading in the memmap file and then reshaping to be square.

    rank_output_path: str or None
        Path to the log file for a single MPI rank. If None, no messages will be printed.
    
    Return: None

    Purpose: Write all similarities belonging to the indices in kernel_indices to the kernel_memmap_path file.
        This is much more efficient than writing similarities one at a time. The np.memmap is read
        all at once and the values are assigned all at once. Because no other MPI rank will have the same
        indices, no clashing should occur.
    '''

    if len(kernel_indices) != len(similarities):
        raise Exception('len(kernel_indices) =', len(kernel_indices), 'should be equal to len(similarities) =',
                        len(similarities), 'but they are not equal.')

    if rank_output_path is not None:
        #rank_print(rank_output_path, 'len(similarities)', len(similarities))
        #rank_print(rank_output_path, 'similarities[:2]', similarities[:2])
        #rank_print(rank_output_path, 'kernel_indices[:2]', kernel_indices[:2])
        #rank_print(rank_output_path, 'type(kernel_indices[0][0])', type(kernel_indices[0][0]))
        start_time = time.time()

    if len(similarities) == 0:
        return

    float32_size = 4
    if num_structures is None:
        fp = np.memmap(kernel_memmap_path, dtype='float32', mode='r+')
        fp_len = len(fp)
        fp.resize((int(np.sqrt(fp_len)), int(np.sqrt(fp_len))))
    else:
        fp = np.memmap(kernel_memmap_path, dtype='float32', mode='r+', shape=(num_structures, num_structures))
        if rank_output_path is not None:
            rank_print(rank_output_path, 'time to load kernel_memmap_path', time.time() - start_time)
    placement_list = np.vstack((kernel_indices[:,0], kernel_indices[:,1])).flatten(), np.vstack((kernel_indices[:,1], kernel_indices[:,0])).flatten()
    if rank_output_path is not None:
        #rank_print(rank_output_path, 'fp.flags', fp.flags)
        start_time = time.time()
    
    fp[placement_list] = similarities + similarities
    if rank_output_path is not None:
        rank_print(rank_output_path, 'time to write to kernel_memmap_path', time.time() - start_time)


def get_krr_task_list(param_path):
    ntrain_paths = set(file_utils.find(param_path, 'ntrain_*', recursive=True))
    # As a quick approximation, assume that if the output file "saved" exists, then this ntrain_path is done. TODO look
    # inside the files to see for sure if all of the ntests were completed to be more sure if this ntrain_path is done or not.
    saved_paths = set([os.path.dirname(p) for p in file_utils.find(param_path, 'saved', recursive=True)])
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
    selection_methods = list(set(eval(inst.get('krr', 'mode'))))
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
    '''
    rank_output_path: str
        Path to the log file for a single MPI rank.

    print_message: anything
        arguments to print()

    Return: None

    Purpose: write to a file that belongs to one MPI rank only
        in order to prevent clashing.
    '''
    with open(rank_output_path, mode='a') as f:
        print(print_message, file=f, flush=True)


def get_traversal_order(kernel_tasks, kernel_calculation_path, rank_output_path):
    '''
    kernel_tasks: np.array shape (num incomplete tasks, 2)
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

    Return: kernel_tasks
    kernel_tasks: np.array shape (num incomplete tasks, 2)
        kernel_tasks is now in the order specified above.
    '''
    # if all structures have the same number of atoms, just return kernel_tasks unmodified
    # 1D array containing the number of atoms in each structure in data_files.
    if not isinstance(kernel_tasks, np.ndarray):
        raise TypeError('kernel_tasks is not of type np.ndarray but it should be. Instead it has the type', type(kernel_tasks))
    num_atoms_arr = file_utils.safe_np_load(os.path.join(kernel_calculation_path, 'num_atoms_arr.npy'), time_frame=0.001, verbose=False, check_file_done_being_written_to=False)
    num_unique_species_arr = file_utils.safe_np_load(os.path.join(kernel_calculation_path, 'num_unique_species_arr.npy'), time_frame=0.001, verbose=False, check_file_done_being_written_to=False)
    if kernel_tasks.shape[0] == 0:
        return kernel_tasks, num_unique_species_arr
    if len(kernel_tasks.shape) != 2:
        raise ValueError('len(kernel_tasks.shape) != 2, but it should be. Instead it has shape', kernel_tasks.shape)
    if kernel_tasks.shape[1] != 2:
        raise ValueError('kernel_tasks.shape[1] != 2, but it should be. Instead it is', kernel_tasks.shape[1])

    if np.all(num_atoms_arr == num_atoms_arr[0]):
        rank_print(rank_output_path, 'returning given incomplete tasks')
        return kernel_tasks, num_unique_species_arr
    else:
        arr_with_idx = np.vstack((np.arange(len(num_atoms_arr)), num_atoms_arr)).T
        c = Counter(num_atoms_arr)
        argsorted_arr = np.array(sorted(arr_with_idx, key=lambda x:(c[x[1]], x[1])))[:,0]
        sorted_order_dct = {i:s for i,s in enumerate(argsorted_arr)}

    rank_print(rank_output_path, 'getting kernel_tasks')

    kernel_tasks = np.array([[int(sorted_order_dct[i]), int(sorted_order_dct[j])] for i,j in kernel_tasks])
    rank_print(rank_output_path, 'got kernel_tasks')
    return kernel_tasks, num_unique_species_arr


def memory_estimate_for_kij(n, l, num_atoms_i, num_atoms_j, num_unique_species_i, num_unique_species_j):
    # units are kB
    kB_per_float = 4.0 / 1024.0

    num_matrix_elements_i = num_atoms_i * (((num_unique_species_i * n)**2) * l + ((num_unique_species_i * n)**2))
    mem_soap_matrix_i = num_matrix_elements_i * kB_per_float
    
    num_matrix_elements_j = num_atoms_j * (((num_unique_species_j * n)**2) * l + ((num_unique_species_j * n)**2))
    mem_soap_matrix_j = num_matrix_elements_j * kB_per_float

    # flattening a matrix using .flatten() doubles its memory usage

    # using tile by x amount (e.g. np.tile(matrix, (x, 1))) increases memory used by about (upperbound) 9000 x + 300000 kilobytes. This
    # was done using a 1000x1000 "matrix" variable which is in the same tier as reasonable sized SOAP atomic matrices
    # (same tier meaning same number of total elements (1000000) and tile appears to allocate in tiers or groups of number
    # of elements. For example, < 300k elements only add less than 5k kilobytes, but 300k - 1000k+ elements add 300k kilobytes and 1*10^8
    # elements add 1.85 million kilobytes. (when x=2))

    # Using repeat by x amount increases memory used by about 4000 x - 7000 kilobytes
    
    # Total memory usage is 
    # (1) storing soap matrix for struct i
    # (2) doing repeat on matrix i with x = num atoms in j and flattening
    # (3) storing repeated array
    # (4) storing soap matrix for struct j
    # (5) doing tile on matrix j with x = num atoms in i and flattening
    # (6) storing tiled array
    # (7) doing dot product of repeated and tiled arrays
    
    # Since these memory events are mostly consecutive, max memory usage is (1) + (4) + max([(2), (5)]) + [(3), (6)][1 - argmax([(2), (5)])]
    # This is because tile comes after repeat so the memory incurred by repeat is 
    # already reliquished by the time the program gets to the tile step, and the extra (non-stored) memory incurred
    # by tile is reliquished by the time the program gets to the np.dot step (and np.dot only takes about 1000 kB
    # additional memory which is much less than that of tile's transient/extra memory). Storing the tiled array is 
    # accounted for in (5) since this is larger than storing the tiled array. Similarly for repeated array. Thus only 1 of (3)
    # or (6) are chosen to contribute to the memory estimate...and the one chosen is the opposite of the one whose transient
    # memory incurred was higher than the other. Example: If (2) incurs more memory than (5), then include (2) and (6)
    # since (3) is absorbed (take into account) by (2).
    storing_repeat_tiled = [num_atoms_j * mem_soap_matrix_i, num_atoms_i * mem_soap_matrix_j]
    predicted_transient_repeat_tiled = [(0.0078 * num_atoms_j - 0.0077) * num_matrix_elements_i + 1500, (0.0078 * num_atoms_i - 0.0078) * num_matrix_elements_j + 1500]
    max_idx = np.argmax(predicted_transient_repeat_tiled)
    predicted_transient = predicted_transient_repeat_tiled[max_idx]
    store_mem = storing_repeat_tiled[1 - max_idx]
    mem_for_struct_pair = mem_soap_matrix_i + mem_soap_matrix_j + predicted_transient + store_mem
    return mem_for_struct_pair


def get_num_ranks_for_kernel_computation(kernel_tasks, n, l, num_atoms_arr, num_unique_species_arr, rank_hostnames, max_mem='all', node_mem=128.0):
    '''
    kernel_tasks: np.array shape (num incomplete tasks, 2)
        Each row is a pair of structure indices that need their similarities evaluated.

    n

    l

    num_atoms_arr

    num_unique_species_arr


    rank_output_path: str
        path to output log file for an MPI rank

    rank_hostnames: list, length: num ranks in comm
        Each row has the rank in columns 0 and that rank's hostname in column 1.

    max_mem: str or float
        If 'all', use all availble memory. If float, use a total of this much memory (possibly useful for
        nodes with a small amount of fast memory). Units GB.

    node_mem: float
        Total RAM on each node. Units GB.

    Return:
        task_list_for_each_rank: list of list
            len(task_list_for_each_rank) = num total ranks in comm. Each sublist is the corresponding kernel
            task list for each rank such that comm.scatter will give the appropriate tasks to each rank.
            Some sublists may be [] if that rank cannot participate due to memory concerns.
            
    Description: We would like to use as many ranks as possible for kernel calculation, but it could
        be possible that if all available ranks load only the vectors necessary for their current
        [i,j] task that the memory could still overflow. Use available memory on the node of the
        root rank as an approximation of the available memory on other nodes in this communicator.
        This is a conservative estimate because the root rank generally as more things loaded
        than other ranks. There is a comm barrier before this function, and a comm.scatter after this
        function such that all ranks on the node of the root rank are idle which makes the estimate
        more accurate.

        You have to treat the problem per node. This is because a node
        may have a task list of large structures and so can only have a small number of ranks whereas as node
        with a task list of small structures can have a relatively large number of ranks.
        Begin by getting memory estimate for each [i,j] pair in kernel task list.
        Divide the total task list evenly amongst the overall number of nodes. For each node,
        Initialize l at 0 and create (num_ranks_per_node - l) partitions to divide this node's tasks into.
        Sum the max mem estimate of each of these partitions. If this sum < available_mem, then we can use
        (num_ranks_per_node - l) ranks on this node, else increment l.

        If all structures are the same size, just get memory usage estimate for a single [i,j] pair.
        min([num ranks on a node, int(available_mem / this memory estimate for K(i,j))]) is the number
        of ranks that can be used per node.
        The memory each rank can use is approximately available_mem / num_ranks_per_node.

        If not all structures are the same size, you have to treat the problem per node. This is because a node
        may have a task list of large structures and so can only have a small number of ranks whereas as node
        with a task list of small structures can have a relatively large number of ranks.
        Begin by getting memory estimate for each [i,j] pair in kernel task list.
        Divide the total task list evenly amongst the overall number of nodes. For each node,
        Initialize l at 0 and create (num_ranks_per_node - l) partitions to divide this node's tasks into.
        Sum the max mem estimate of each of these partitions. If this sum < available_mem, then we can use
        (num_ranks_per_node - l) ranks on this node, else increment l.
    '''
    unique_hostnames = list(set(rank_hostnames))
    if max_mem == 'all':
        # kB units
        # Assume all nodes have the same or greater available memory as the node with the root rank.
        available_mem = dict(psutil.virtual_memory()._asdict())['available'] / 1024.0
    else:
        available_mem = max_mem  * (1024 ** 2) - (node_mem * (1024 ** 2) - dict(psutil.virtual_memory()._asdict())['available'] / 1024.0)
    
    kernel_tasks_for_hostnames = list_utils.split_up_list_evenly(kernel_tasks, len(unique_hostnames))
    task_list_for_each_rank = [[] for rank in range(len(rank_hostnames))]
    for hostname_i,hostname in enumerate(unique_hostnames):
        kernel_tasks_for_this_hostname = kernel_tasks_for_hostnames[hostname_i]
        struct_idxs_with_this_hostname = list(set(np.array(kernel_tasks_for_this_hostname).flatten()))
        ranks_with_this_hostname = np.where(rank_hostnames == hostname)[0]
        num_ranks_with_this_hostname = len(ranks_with_this_hostname)
        num_atoms_arr_for_this_hostname = num_atoms_arr[struct_idxs_with_this_hostname]
        num_unique_species_arr_for_this_hostname = num_unique_species_arr[struct_idxs_with_this_hostname]
        
        if np.all(num_atoms_arr_for_this_hostname == num_atoms_arr_for_this_hostname[0]) and np.all(num_unique_species_arr_for_this_hostname == num_unique_species_arr_for_this_hostname[0]):
            mem_for_struct_pair = memory_estimate_for_kij(n, l, num_atoms_arr_for_this_hostname[0], num_atoms_arr_for_this_hostname[0], num_unique_species_arr_for_this_hostname[0], num_unique_species_arr_for_this_hostname[0])
            #print('mem_for_struct_pair', mem_for_struct_pair)
            #print('num_ranks_with_this_hostname', num_ranks_with_this_hostname)
            #print('available_mem', available_mem)
            num_ranks_allowed = min([num_ranks_with_this_hostname, int(available_mem / mem_for_struct_pair)])
            #print('num_ranks_allowed', num_ranks_allowed)
        else:
            # Get memory estimate for each task assigned to this node
            task_mem_estimates = [memory_estimate_for_kij(n, l, num_atoms_arr[kernel_task[0]], num_atoms_arr[kernel_task[1]], num_unique_species_arr[kernel_task[0]], num_unique_species_arr[kernel_task[1]])
                    for kernel_task in kernel_tasks_for_this_hostname]
            
            num_partitions = num_ranks_with_this_hostname
            while num_partitions > 0:
                split_task_mem_estimates = list_utils.split_up_list_evenly(task_mem_estimates, num_partitions)
                max_simultaneous_mem_by_all_ranks_on_this_node = sum([max(partition_mem_estimates) for partition_mem_estimates in split_task_mem_estimates])
                if available_mem > max_simultaneous_mem_by_all_ranks_on_this_node:
                    print('num_ranks_allowed', num_ranks_allowed)
                    print('split_task_mem_estimates', split_task_mem_estimates)
                    num_ranks_allowed = num_partitions
                    break
                num_partitions -= 1
            if num_partitions <= 0:
                raise Exception('It appears that you would not have enough memory for even one rank to store enough memory for one pair of structures at a time',
                        'available_mem =', available_mem, 'max_simultaneous_mem_by_all_ranks_on_this_node =', max_simultaneous_mem_by_all_ranks_on_this_node)

        #Actually, I need to return the list of lists where each sublist is a task list for each rank. The list must be
        # in order such that comm.scatter will work. The task list will be empty for ranks that are above the allowed
        # number of ranks on a node - nontheless, an empty list should still be there for placeholder purposes (again,
        # so that the comm.scatter will work).
        split_kernel_tasks = list_utils.split_up_list_evenly(kernel_tasks_for_this_hostname, num_ranks_allowed)
        for rank_i,rank in enumerate(ranks_with_this_hostname[:num_ranks_allowed]):
            task_list_for_each_rank[rank] = split_kernel_tasks[rank_i]

    return task_list_for_each_rank


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
    

def load_soaps(task_idx, loaded_soaps, task_list, num_atoms_arr, global_envs_dir, lowmem, lowestmem, breathing_room_factor, rank_output_path,
        kernel_calculation_path, krrtest=False):
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
    rank_output_path: str
        Path for this MPI rank to write to its own file for log purposes.

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
    
    num_atoms_in_i = int(num_atoms_arr[i])
    num_atoms_in_j = int(num_atoms_arr[j])
    #rank_print(rank_output_path, 'num_atoms_in_i', num_atoms_in_i)
    matrices_to_keep = ['atomic_envs_matrix_' + str(i) + '_repeat_' + str(num_atoms_in_j), 'atomic_envs_matrix_' + str(j) + '_tile_' + str(num_atoms_in_i)]
    delete_all_unnecessary_matrices(loaded_soaps, matrices_to_keep)
    #rank_print(rank_output_path, 'deleted unnecessary matrices')
    for matrix_to_keep in matrices_to_keep:
        if matrix_to_keep not in loaded_soaps:
            if krrtest and 'tile' in matrix_to_keep:
                fpath = os.path.join(kernel_calculation_path, 'tmpstructures', matrix_to_keep + '.npy')
            else:
                fpath = os.path.join(global_envs_dir, matrix_to_keep + '.npy')
            #rank_print(rank_output_path, 'fpath', fpath)
            loaded_soaps[matrix_to_keep] = file_utils.safe_np_load(fpath, time_frame=0.001, verbose=False, check_file_done_being_written_to=False)
    #rank_print(rank_output_path, 'num_atoms_in_j', num_atoms_in_j)
    #Could insert a statement here to delete loaded_soaps[str(j)] if it's not needed in any future calculations to further save on memory
    #Could actually see if a particular matrix will be used later on and prefer to keep it if you could instead delete one that will not be used later on
    return num_atoms_in_i, num_atoms_in_j, matrices_to_keep


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


def write_num_atoms_arr_and_num_unique_species_arr(wdir_path, data_files):
    '''
    wdir_path: str
        Directory to write the these files to.
    data_files: list of str
        List of paths to .json files, each with a structure.

    Purpose: write a .npy file which is a single column
        listing the number of atoms of each structure. Also,
        write a .npy file which is a single column
        listing the number of unique atomic species of each structure.
    '''
    num_atoms_arr_outfpath = os.path.join(wdir_path, 'num_atoms_arr.npy')
    num_unique_species_arr_outfpath = os.path.join(wdir_path, 'num_unique_species_arr.npy')
    if os.path.exists(num_unique_species_arr_outfpath):
        # This file already exists so this run must be restarting.
        return np.load(num_unique_species_arr_outfpath)
    
    num_atoms_lst = []
    num_unique_species_lst = []
    for json_file in data_files:
        struct = file_utils.get_dct_from_json(json_file, load_type='load')
        if 'properties' not in struct:
            raise ValueError('json structure file must have a "properties" key')
        if 'geometry' not in struct:
            raise ValueError('json structure file must have a "geometry" key')
        geometry = np.array([struct['geometry'][i][:3] for i in range(len(struct['geometry']))])
        napc = len(geometry)
        num_atoms_lst.append(napc)
        num_unique_species_lst.append(len(set([struct['geometry'][i][3] for i in range(len(struct['geometry']))])))
    np.save(num_atoms_arr_outfpath, np.array(num_atoms_lst, dtype='float32'))
    num_unique_species_arr = np.array(num_unique_species_lst, dtype='float32')
    np.save(num_unique_species_arr_outfpath, num_unique_species_arr)
    return np.float32(num_unique_species_arr)


def write_en_dat(wdir_path, data_files, out_fname, single_molecule_energy, energy_name='energy'):
    '''
    wdir_path: str
        Directory to write the energy dat file to.
    data_files: list of str
        List of paths to .json files, each with a structure.
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
    num_unique_species_arr_outfpath = os.path.join(wdir_path, 'num_unique_species_arr.npy')
    if os.path.exists(num_atoms_arr_outfpath):
        # This file already exists so this run must be restarting.
        return
    
    num_atoms_lst = []
    num_unique_species_lst = []
    with open(out_fpath, mode='w') as f:
        for json_file in data_files:
            struct = file_utils.get_dct_from_json(json_file, load_type='load')
            if 'properties' not in struct:
                raise ValueError('json structure file must have a "properties" key')
            if 'geometry' not in struct:
                raise ValueError('json structure file must have a "geometry" key')
            geometry = np.array([struct['geometry'][i][:3] for i in range(len(struct['geometry']))])
            napc = len(geometry)
            num_atoms_lst.append(napc)
            num_unique_species_lst.append(len(set([struct['geometry'][i][3] for i in range(len(struct['geometry']))])))
            if energy_name in struct['properties']:
                en=float(struct['properties'][energy_name])
                if 'nmpc' in struct['properties']:
                    nmpc = int(struct['properties']['nmpc'])
                elif 'Z' in struct['properties']:
                    nmpc = int(struct['properties']['Z'])
                else:
                    raise Exception('Must have "nmpc" or "Z" as a property in your structure files indicating the number of molecules per unit cell')
                energy = normalized_BE_by_napc(napc, nmpc, en, single_molecule_energy)
            else:
                energy = 'none'
            f.write(str(energy) + '\n')
    num_atoms_arr = np.array(num_atoms_lst, dtype='float32')
    np.save(num_atoms_arr_outfpath, num_atoms_arr)
    np.save(num_unique_species_arr_outfpath, np.array(num_unique_species_lst, dtype='float32'))
    return num_atoms_arr


def get_params_in_progress(soap_runs_dir, param_symbols=['n', 'l', 'c', 'g', 'zeta']):
    '''
    soap_runs_dir: str
        Path of the soap_runs directory which contains parameter directories e.g.
        n8-l8-c4.0-g0.7-zeta1.0

    param_symbols: list of str
        List of designations for each parameter found in the directory names. A list
        of lists each of length len(param_symbols) will be returned. 

    Return: params_lst
    params_lst: list of list of number
        list of parameter sets that are currently in soap_runs_dir

    Purpose: When only wanting to restart the parameter sets in soap_runs_dir,
        retrieve which parameter sets those are by looking in soap_runs_dir
        and parsing the directory names for the parameter sets.
    '''
    param_dirs = file_utils.find(soap_runs_dir, '*zeta*', recursive=False)
    params_lst = []
    for param_dir in param_dirs:
        split_param_dir = os.path.basename(param_dir).split('-')
        param_set = []
        for param_symbol in param_symbols:
            for i in range(len(split_param_dir)):
                if param_symbol in split_param_dir[i]:
                    param_value = split_param_dir[i].split(param_symbol)[-1]
                    if param_symbol in ['n', 'l']:
                        param_value = int(param_value)
                    else:
                        param_value = float(param_value)
                    param_set.append(param_value)
                    break
        params_lst.append(param_set)
    return params_lst


def param_set_complete(param_path, selection_methods, kernel_calculation_path, params_set):
    '''
    param_path: str
        Path to parameter set under soap_runs dir. e.g. 
        soap_runs/n8-l8-c4.0-g0.7-zeta1.0
    
    selection_methods: list of str
        List of selection (sampling) methods that you will use to select the
        training set. e.g. iss, fibs, random, fps, cur

    kernel_calculation_path: str
        Path of dir that the kernel will be in once calculated.

    params_set: list of numbers
        List of parameters used. n, l, c, g, zeta = params_set

    Return: bool
        True: kernel is calculated and krr is done for all selection methods
        False: o/w

    Purpose: When restarting, determine if a particular parameter set has already
        been calculated - both the kernel and krr for all desired selection 
        methods.
    '''
    if not os.path.exists(os.path.join(kernel_calculation_path, 'completed_kernel')):
        return False

    if not os.path.exists('krr_results.dat'):
        return False
    # Don't bother with runs that had bad results with other krr selection methods
    fp = np.memmap('krr_results.dat', dtype='float32', mode='r')
    fp_len = len(fp)
    fp.resize((int(fp_len / 6), 6))
    for row in fp:
        if np.allclose(row[:-1], params_set) and row[-1] < 0.5:
            return True
    
    selection_methods = selection_methods
    complete_selection_methods = [path.split('/')[1] for path in file_utils.find(param_path, 'saved')]

    for selection_method in selection_methods:
        if selection_method not in complete_selection_methods:
            return False

    return True

def get_krr_test_task_list(num_test_structs, num_structs_in_kernel):
    '''
    num_test_structs: int
        This is the number of structures that you want to predict the property for.

    num_structs_in_kernel: int
        This is the number of structures in the kernel under kernel_calculation_path

    Return: overall_krr_test_task_list
        overall_krr_test_task_list: np.array, shape (x, 2)
            Pairs of indices of structures to get the soap similarity between. The
            first len(kernel) of indices belong to the structures in the
            kernel. Their environments are found in join(kernel_calculation_path,
            tmpstructures). The last num_test_structs of indices belong to the
            structures you want to predict the property for. e.g. If there
            were 3 structures in the kernel and num_test_structs was 2, then
            the index assigned to the structures in the kernel are 0, 1, 2 and
            the index assigned to the test structures are 3, 4.

    Purpose: get the array of pairs of structure indices which must have their
        soap similarity calculated and put into a rectangular matrix in order to
        perform krr. The 
    '''
    return np.array([[j, i] for j in range(num_structs_in_kernel,num_structs_in_kernel + num_test_structs) for i in range(num_structs_in_kernel)])


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
    restart_only = eval(inst.get_with_default('master', 'restart_only', False))
    c_precision = eval(inst.get_with_default('calculate_kernel', 'c_precision', 0.5))
    g_precision = eval(inst.get_with_default('calculate_kernel', 'g_precision', 0.5))
    zeta_precision = eval(inst.get_with_default('calculate_kernel', 'zeta_precision', 0.25))
    prec_dct = {'c': c_precision, 'g': g_precision, 'zeta': zeta_precision}
    params_to_get = ['n','l','c','g','zeta']
    sname = 'krr'
    krr_results_memmap_offset = 0
    float32_size = 4
    #original working dir
    owd = os.getcwd()
    krr_results_memmap_fpath = os.path.join(owd,'krr_results.dat')
    soap_runs_dir = os.path.join(owd, 'soap_runs')
    num_param_combos = eval(inst.get('master', 'num_param_combos_per_replica'))

    #root_print(comm_world.rank, 'about to get output fpath')
    #overall_outfile_fpath = file_utils.get_stdout_fpath()
    #root_print(comm_world.rank, 'overall_outfile_fpath', overall_outfile_fpath)
    overall_outfile_fpath = os.path.join(owd, 'output.out')
    
    if num_param_combos is None:
        if restart_only:
            iterable = get_params_in_progress(soap_runs_dir)
        else:
            params_list = [inst.get_list('calculate_kernel', p) for p in params_to_get]
            params_combined_iterable = set(itertools.product(*params_list))
            iterable = params_combined_iterable
    else:
        iterable = range(num_param_combos)
    if comm_world.rank == 0 and num_param_combos is not None:
        # We're using all Integer Dimension objects because it will reduce the search space, however, we want 0.5 level accuracy for now on
        # c, g, and zeta so I'm multiplying their ranges by 2 to get a value, and then later dividing that value by 2 to get the true value to 
        # pass to the kernel.
        n_lower_bound = eval(inst.get_with_default('calculate_kernel', 'n', [4]))[0]
        n_upper_bound = eval(inst.get_with_default('calculate_kernel', 'n', [14]))[-1]
        l_lower_bound = eval(inst.get_with_default('calculate_kernel', 'l', [4]))[0]
        l_upper_bound = eval(inst.get_with_default('calculate_kernel', 'l', [14]))[-1]
        c_lower_bound = eval(inst.get_with_default('calculate_kernel', 'c', [3]))[0]
        c_upper_bound = eval(inst.get_with_default('calculate_kernel', 'c', [30]))[-1]
        g_lower_bound = eval(inst.get_with_default('calculate_kernel', 'g', [0.5]))[0]
        g_upper_bound = eval(inst.get_with_default('calculate_kernel', 'g', [40]))[-1]
        zeta_lower_bound = eval(inst.get_with_default('calculate_kernel', 'zeta', [0.75]))[0]
        zeta_upper_bound = eval(inst.get_with_default('calculate_kernel', 'zeta', [1.25]))[-1]

        print('n_lower_bound = {}, n_upper_bound = {}, l_lower_bound = {}, l_upper_bound = {}, c_lower_bound = {}, c_upper_bound = {}, g_lower_bound = {}, g_upper_bound = {}, zeta_lower_bound = {}, zeta_upper_bound = {},'.format(n_lower_bound, n_upper_bound, l_lower_bound, l_upper_bound, c_lower_bound, c_upper_bound, g_lower_bound, g_upper_bound, zeta_lower_bound, zeta_upper_bound), flush=True)
        print('c_precision = {}, g_precision = {}, zeta_precision = {}'.format(c_precision, g_precision, zeta_precision), flush=True)
        Int_vars = [
            [int(n_lower_bound), int(n_upper_bound)],
            [int(l_lower_bound), int(l_upper_bound)],
            [int(c_lower_bound / c_precision), int(c_upper_bound / c_precision)],
            [int(g_lower_bound / g_precision), int(g_upper_bound / g_precision)],
            [int(zeta_lower_bound / zeta_precision), int(zeta_upper_bound / zeta_precision)]]

        for bounds in Int_vars:
            if bounds[0] > bounds[1]:
                raise ValueError('Cannot have lower bound be higher than upper bound', bounds, Int_vars)

        # indices in Int_vars of variables that have equal lower and upper bounds meaning they should
        # not be varied but always use the value given by lower = upper bound
        single_val_int_var_idx = [i for i,bounds in enumerate(Int_vars) if bounds[0] == bounds[1]]
        single_vals_int_var = [bounds[0] for bounds in Int_vars if bounds[0] == bounds[1]]
        Int_vars_for_opt = [bounds for i,bounds in enumerate(Int_vars) if bounds[0] != bounds[1]]

        opt = Optimizer(Int_vars_for_opt, n_initial_points=5)
        opt.dct = {}

        # Read in parameters stored in krr_results_memmap_fpath into opt if they aren't already there. This is
        # a part of the restart procedure.
        if os.path.exists(krr_results_memmap_fpath):
            krr_results_memmap = np.memmap(krr_results_memmap_fpath, dtype='float32', mode='r')
            krr_results_memmap.resize((int(len(krr_results_memmap) / 6), 6))
 
            pts_from_krr_results = tuple(map(tuple, krr_results_memmap[:,:-1]))
            output_values_from_krr_results = list(krr_results_memmap[:,-1])
            pt_set_from_krr_results = set(pts_from_krr_results)
            dct_pt_set = set(opt.dct.keys())
            novel_pts = pt_set_from_krr_results - dct_pt_set
            output_values = [output_values_from_krr_results[i] for i,pt in enumerate(pts_from_krr_results) if pt in novel_pts]
            pts_to_tell = list(map(list, novel_pts))
            for i,pt in enumerate(pts_to_tell):
                try:
                    opt = tell_opt(opt, [pt], [output_values[i]])
                except ValueError:
                    # This is usually due to having changed the parameter space bounds in between runs, so it isn't a problem.
                    continue
            del krr_results_memmap
        
        search_space_size = get_search_space_size(Int_vars)
    
    root_print(comm_world.rank, 'using ' + str(comm.size) + ' MPI ranks on ' + str(params.num_structures_to_use) + ' structures.')
        
    data_files, data_files_idx = put_struct_dir_pct_in_arr(params.test_struct_dir_pct_lst, params.train_struct_dir_pct_lst, params.structure_dirs, params.num_structures_to_use, data_files_matrix=None)

    start_time = time.time()

    for params_iter in iterable:
        start_time_param = time.time()
        param_string = ''
        if num_param_combos is None:
            params_set = params_iter
        else:
            if comm_world.rank == 0:
                params_sets = ask_opt(opt, Int_vars_for_opt, search_space_size, num_novel_pts_to_get=num_replicas, max_iterations=100, strategy='cl_min')
                root_print(comm_world.rank, 'params_sets gotten when asked:', params_sets)
                params_sets = [[math_utils.round(p,3,leave_int=True) for p in param_set] for param_set in params_sets]

                for i,param_set in enumerate(params_sets):
                    params_sets[i] = list_utils.multi_put(param_set, single_val_int_var_idx, single_vals_int_var, append_if_beyond_length=False)

                root_print(comm_world.rank, 'rounded params_sets gotten when asked and now including single value params: [n,l,c,g,zeta]', params_sets)
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
        n = math_utils.round(n, 3, leave_int=True)
        l = math_utils.round(l, 3, leave_int=True)
        c = math_utils.round(c, 3, leave_int=False)
        g = math_utils.round(g, 3, leave_int=False)
        zeta = math_utils.round(zeta, 3, leave_int=False)
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
        kernel_calculation_path = os.path.abspath(os.path.join(param_path, 'calculate_kernel'))
        params.kernel_calculation_path = kernel_calculation_path

        if param_set_complete(param_path, params.selection_methods, params.kernel_calculation_path, params_set):
            continue

        root_print(comm_world.rank, 'param_path', param_path)
        
        rank_output_dir = os.path.join(param_path, 'output_log')
        rank_output_path = os.path.join(rank_output_dir, 'output_from_rank_' + str(comm.rank) + '.out')

        if comm.rank == 0 and restart_only:
            #Clear out old log files
            if os.path.exists(rank_output_dir):
                root_print(comm.rank, 'rank_output_dir', rank_output_dir, 'exists but we are restarting so remove it.')
                file_utils.rm(rank_output_dir)

        #mpi_utils.parallel_mkdir(comm.rank, param_path, time_frame=0.001)
        mpi_utils.parallel_mkdir(comm.rank, rank_output_dir, time_frame=0.001)
        mpi_utils.parallel_mkdir(comm.rank, kernel_calculation_path, time_frame=0.001)

        en_all_dat_fname = inst.get('krr', 'props')
        
        single_molecule_energy = eval(inst.get('master', 'single_molecule_energy'))

        kernel_complete = os.path.exists(os.path.join(kernel_calculation_path, 'completed_kernel'))
        kernel_memmap_path = os.path.join(kernel_calculation_path, 'kernel.dat')
        global_envs_dir = os.path.join(kernel_calculation_path, 'tmpstructures')
        en_all_dat_fpath = os.path.join(kernel_calculation_path, en_all_dat_fname)
        if not kernel_complete:
            if comm.rank == 0:
                num_atoms_arr = write_en_dat(kernel_calculation_path, data_files, en_all_dat_fname, single_molecule_energy)

            # Calculate global environments if not already calculated
            
            mpi_utils.parallel_mkdir(comm.rank, global_envs_dir, time_frame=0.001)
            if comm.rank != 0:
                num_atoms_arr = file_utils.safe_np_load(os.path.join(kernel_calculation_path, 'num_atoms_arr.npy'), time_frame=0.001, verbose=False, check_file_done_being_written_to=True)

            start_time_envs = time.time()
            num_structures = len(file_utils.get_lines_of_file(en_all_dat_fpath))
            
            root_print(comm_world.rank, str(num_structures) + ' structures are in the pool.')
            # make an array that will store the number of atoms in every structure.
            # Must have at least 2 ranks so that 1 can be the master rank
            ##print('num_structures', num_structures, flush=True)
            ##print('comm.size', comm.size, flush=True)
            
            global_envs_incomplete_tasks = np.arange(num_structures)
            global_envs_incomplete_tasks = mpi_utils.split_up_list_evenly(global_envs_incomplete_tasks, comm.rank, comm.size)
            rank_print(rank_output_path, 'my assigned global environments to calculate:', global_envs_incomplete_tasks)

            alchem = alchemy(mu=params.soap_standalone_options['mu'])
            os.chdir(kernel_calculation_path)
            
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
                rank_print(rank_output_path, 'Using parameters', 'c = {}, cotw = {}, n = {}, l = {}, g = {}, cw = {}, zeta = {}, nocenter = {}, exclude = {}, unsoap = {}'.format(c, params.soap_standalone_options['cotw'], n, l, g, params.soap_standalone_options['cw'], zeta, params.soap_standalone_options['nocenter'], params.soap_standalone_options['exclude'], params.soap_standalone_options['unsoap']))
                for at in al:
                    si = structure(alchem)
                    si.parse(at, c, params.soap_standalone_options['cotw'], n, l, g, params.soap_standalone_options['cw'], params.soap_standalone_options['nocenter'], params.soap_standalone_options['exclude'], unsoap=params.soap_standalone_options['unsoap'], kit=params.soap_standalone_options['kit'])
                    sl.append(si.atomic_envs_matrix, num_atoms_arr, params.user_num_atoms_arr, store_tiled=True)
                rank_print(rank_output_path, 'time to compute envs for {} structures: {}'.format(stop - start, time.time() - just_env_start_time))
                del global_envs_incomplete_tasks
                del si
                del sl
                del al
            start_time_kernel = time.time()
            rank_print(rank_output_path,'time to read input structures and calculate environments', start_time_kernel - start_time_envs)
            root_print(comm_world.rank,'time to read input structures and calculate environments', start_time_kernel - start_time_envs)
            
            # Calculate kernel if not already calculated
            
            # Not supporting restarts of kernel b/c it is so much slower to write one similarity at a time and also requires a comm.barrier.
            # However, we should check if the kernel has been completed which we'll know if the root rank made a file called completed_kernel
            if comm.rank == 0:
                kernel_tasks_rows, kernel_tasks_cols = np.triu_indices(num_structures)
                kernel_tasks = np.array(list(zip(kernel_tasks_rows, kernel_tasks_cols)))
                del kernel_tasks_rows
                del kernel_tasks_cols
                
                fp = np.memmap(kernel_memmap_path, dtype='float32', mode='w+', shape=(num_structures, num_structures))
                del fp
                root_print(comm.rank, 'getting kernel traversal order')
                kernel_tasks, num_unique_species_arr = get_traversal_order(kernel_tasks, kernel_calculation_path, rank_output_path)
                full_task_list = deepcopy(kernel_tasks)
            else:
                kernel_tasks = None

            # Barrier to get a better available memory estimate and doesn't hurt too much because we do a comm.scatter afterwards
            comm.barrier()
            if comm.rank == 0:
                get_rank_tasks_start_time = time.time()
                num_ranks_for_nodes = get_num_ranks_for_kernel_computation(kernel_tasks, n, l, num_atoms_arr, num_unique_species_arr, rank_hostnames, params.max_mem, params.node_mem)
                rank_print(rank_output_path, 'num_ranks_for_nodes time', time.time() - get_rank_tasks_start_time)    
            else:
                num_ranks_for_nodes  = None
            
            scatter_start_time = time.time()
            kernel_tasks = comm.scatter(num_ranks_for_nodes, root=0)
            rank_print(rank_output_path, 'kernel_tasks scatter time', time.time() - scatter_start_time)

            rank_print(rank_output_path, 'For kernel computation, got {} kernel_tasks'.format(len(kernel_tasks))) #, kernel_tasks)
            loaded_soaps = {}
            # kernel_tasks should be traversal order of the kernel matrix tasks such that it should be a np.array with shape (num_tasks_per_rank, 2) where
            # each row has (as the first element) the struct index of a structure which should get its atomic env matrix repeated and (as the second element)
            # the struct index of a structure which should get its atomic env matrix tiled.
            # kernel_tasks is np.array with shape usually being (num_tasks_per_rank, 2) where each row is a pair of structures that need their similarities evaluated.
            breathing_room_factor = ranks_per_node * breathing_room_factor_per_node
            just_kernel_start_time = time.time()
            my_kernel_data = []
            root_print(comm.rank, 'Beginning kernel calculation')
            rank_print(rank_output_path, 'Beginning kernel calculation')
            for task_idx in range(len(kernel_tasks)):
                if params.verbose:
                    rank_print(rank_output_path, 'computing task_idx for kernel', task_idx)
                num_atoms_in_i, num_atoms_in_j, matrices_to_keep = load_soaps(task_idx, loaded_soaps, kernel_tasks, num_atoms_arr, global_envs_dir, params.lowmem, params.lowestmem, breathing_room_factor, rank_output_path, kernel_calculation_path, krrtest=False)
                i,j = kernel_tasks[task_idx]
                sij = np.dot(loaded_soaps[matrices_to_keep[0]], loaded_soaps[matrices_to_keep[1]]) ** zeta
                my_kernel_data.append(sij)
            rank_print(rank_output_path, 'time to compute kernel entries for {} entries: {}'.format(len(kernel_tasks), time.time() - just_kernel_start_time))
            write_kernel_start_time = time.time()
            kernel_data = comm.gather(my_kernel_data, root=0)
            rank_print(rank_output_path,'time to gather kernel_data', time.time() - write_kernel_start_time)
            if comm.rank == 0:
                kernel_data = list_utils.flatten_list(kernel_data)
                write_similarities_to_file(full_task_list, kernel_data, kernel_memmap_path, num_structures=num_structures, rank_output_path=rank_output_path)
                rank_print(rank_output_path,'wrote kernel_data')
                
            rank_print(rank_output_path,'ending kernel calculation step.')
            end_time_kernel = time.time()
            if comm.rank == 0:
                rank_print(rank_output_path,'time to calculate kernel', end_time_kernel - start_time_kernel)
                root_print(comm_world.rank,'time to calculate kernel', end_time_kernel - start_time_kernel)
                rank_print(rank_output_path,'time to gather and write kernel', end_time_kernel - write_kernel_start_time)
                file_utils.write_to_file(os.path.join(kernel_calculation_path, 'completed_kernel'), 'completed_kernel', mode='w')
                del kernel_data
                del full_task_list
            del my_kernel_data
            del kernel_tasks
        start_time_krr = time.time()
        if 'krr' in params.process_list:
            # barrier here to ensure kernel is completed before beginning krr.
            if comm.rank == 0:
                #create krr working directories (ntrain path)
                create_krr_dirs(inst, param_path)
            comm.barrier()
            rank_print(rank_output_path, 'entering krr')

            overall_krr_task_list = get_krr_task_list(param_path)

            if len(overall_krr_task_list) > 0:
            
                krr_task_list = mpi_utils.split_up_list_evenly(overall_krr_task_list, comm.rank, comm.size)

                outfile_path = inst.get(sname, 'outfile')
                props_fname = params.krr_param_list[params.krr_param_list.index('--props') + 1]
                props_path = os.path.join(kernel_calculation_path, os.path.basename(os.path.abspath(props_fname)))
                
                # krr_task_list is np.array with shape (,number of tasks) where each element is an ntrain path.
                rank_print(rank_output_path, 'krr_task_list recieved in krr', krr_task_list)
                just_krr_start_time = time.time()
                my_krr_results = []
                krr_results_list = []
                for ntrain_path in krr_task_list:
                    rank_print(rank_output_path, 'doing krr for ntrain_path', ntrain_path)
                    one_krr_start_time = time.time()
                    os.chdir(ntrain_path)
                    ntrain, ntest, selection_method = get_specific_krr_params(ntrain_path)
                    # do krr
                    sys.stdout = open(outfile_path, 'w')
                    avg_test_R2 = krr.main(kernels=[kernel_memmap_path], props=[props_path], kweights=params.krr_standalone_options['kweights'], mode=selection_method, ntrain=ntrain,
                                    ntest=ntest, ntrue=params.krr_standalone_options['ntrue'], csi=params.krr_standalone_options['csi'],
                                    sigma=params.krr_standalone_options['sigma'], ntests=params.krr_standalone_options['ntests'], 
                                    savevector=params.krr_standalone_options['saveweights'], refindex=params.krr_standalone_options['refindex'],
                                    inweights=params.krr_standalone_options['pweights'])
                    rank_print(rank_output_path, 'Completed krr for above ntrain_path')
                    my_krr_results.append(params_set + [avg_test_R2])
                    rank_print(rank_output_path, 'time to compute one krr with selection method {} is: {}'.format(selection_method, time.time() - one_krr_start_time))
                end_time_krr = time.time()
                sys.stdout = open(overall_outfile_fpath, 'a')
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
                    krr_results_matrix = np.array(list_utils.flatten_list(krr_results))
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
                        overall_krr_results_matrix[:,2] = np.int64(overall_krr_results_matrix[:,2] / c_precision)
                        overall_krr_results_matrix[:,3] = np.int64(overall_krr_results_matrix[:,3] / g_precision)
                        overall_krr_results_matrix[:,4] = np.int64(overall_krr_results_matrix[:,4] / zeta_precision)
                        #root_print(comm_world.rank, 'overall_krr_results_matrix after prec', overall_krr_results_matrix)
                        #root_print(comm_world.rank, 'list(map(list, np.int64(np.around(overall_krr_results_matrix[:,:-1]))))', np.int64(np.around(list(map(list, overall_krr_results_matrix[:,:-1])))))
                        # Used np.int64 since we're using all integers in the param space.

                        krr_results_values = overall_krr_results_matrix[:,-1]
                        krr_results_parameters = overall_krr_results_matrix[:,:-1]
                        # Can't tell opt parameters that aren't in the space (which are ones that don't vary so we couldn't include them)
                        krr_results_parameters_to_tell_opt = list_utils.multi_delete(krr_results_parameters, single_val_int_var_idx, axis=1)

                        opt = tell_opt(opt, list(map(list, krr_results_parameters_to_tell_opt)), list(krr_results_values))
                    
            elif comm.rank == 0 and len(other_root_ranks) > 0:
                comm_world.send(krr_results_list, dest=0, tag=2)
        else:
            if num_param_combos is None:
                rank_print(rank_output_path, 'no krr; moving on.')
            else:
                rank_print(rank_output_path, 'no krr; Cant get R2 so cant get new param. Returning.')
                return
        if 'krr_test' in params.process_list:
            sname = 'krr_test'
            for test_i, test_structs_dir in enumerate(params.test_structs_dirs):
                krr_test_start_time = time.time()
                num_test_structs = params.num_test_structs_list[test_i]
                if comm.rank == 0:
                    overall_krr_test_task_list = get_krr_test_task_list(num_test_structs, params.num_structures_to_use)
                krr_test_dir = os.path.join(param_path, 'krr_test')
                krr_test_dirs = file_utils.find(krr_test_dir, 'krr_test_*', recursive=False)
                # No restarts so that one can use the same kernel for different test sets.
                # The program will see which x has the largest value of krr_test_x dir which has already completed from previous runs
                x = len(krr_test_dirs)
                if x != 0:
                    possible_current_krr_test_dir = os.path.join(krr_test_dir, 'krr_test_' + str(x))
                    if not os.path.exists(possible_current_krr_test_dir):
                        x -= 1
                        possible_current_krr_test_dir = os.path.join(krr_test_dir, 'krr_test_' + str(x))
                        # Is the previous dir complete?
                        if os.path.exists(os.path.join(possible_current_krr_test_dir, 'tmpstructures')):
                            x += 1
                current_krr_test_dir = os.path.join(krr_test_dir, 'krr_test_' + str(x))
                mpi_utils.parallel_mkdir(comm.rank, current_krr_test_dir)
                num_structs_in_kernel = params.num_structures_to_use
                # Use krr to get the weights of the overall kernel
                props_path = os.path.join(kernel_calculation_path, inst.get(sname, 'props'))
                kernel_weights_fpath = os.path.join(kernel_calculation_path, file_utils.fname_from_fpath(params.krr_test_standalone_options['saveweights']) + '.npy')
                if not os.path.exists(kernel_weights_fpath):
                    os.chdir(kernel_calculation_path)
                    krr.main(kernels=[kernel_memmap_path], props=[props_path], kweights=params.krr_test_standalone_options['kweights'], mode='krr_test', ntrain=num_structs_in_kernel,
                            ntest=num_test_structs, ntrue=params.krr_test_standalone_options['ntrue'], csi=params.krr_test_standalone_options['csi'],
                            sigma=params.krr_test_standalone_options['sigma'], ntests=1, 
                            savevector=params.krr_test_standalone_options['saveweights'], refindex=params.krr_test_standalone_options['refindex'],
                            inweights=params.krr_test_standalone_options['pweights'])
    
                # Get the atomic environment matrices for the test structures and index them starting a i = num_structs_in_kernel
                rank_print(rank_output_path, 'entering ' + current_krr_test_dir)
                os.chdir(current_krr_test_dir)
                # Calculate global environments if not already calculated
                global_envs_dir = os.path.join(current_krr_test_dir, 'tmpstructures')
                mpi_utils.parallel_mkdir(comm.rank, global_envs_dir, time_frame=0.001)
    
                start_time_test_envs = time.time()
                
                root_print(comm_world.rank, 'Creating atomic environment matrices for ' + str(num_test_structs) + ' structures.')
                
                # The program will recalculate environments that were already calculated (applicable if restarting the calculation) because this part is could be faster than the comm.barrier that would otherwise be required.
                global_envs_incomplete_tasks = np.arange(num_structs_in_kernel, num_structs_in_kernel + num_test_structs)
                global_envs_incomplete_tasks = mpi_utils.split_up_list_evenly(global_envs_incomplete_tasks, comm.rank, comm.size)
                rank_print(rank_output_path, 'my assigned global environments to calculate:', global_envs_incomplete_tasks)
    
                alchem = alchemy(mu=params.soap_standalone_options['mu'])
                
                # global_envs_incomplete_tasks is a np.array of tasks where each task is the index in the structure list of a structure that still
                # needs its global soap descriptor calculated.
                # Calculate the global soap descriptor for each structure whose index is in global_envs_incomplete_tasks
                if len(global_envs_incomplete_tasks) > 0:
                    start = global_envs_incomplete_tasks[0] - num_structs_in_kernel
                    rank_print(rank_output_path, 'Beginning computation of global envs with start =', start)
                    if global_envs_incomplete_tasks[-1] == start:
                        stop = start + 1
                    else:
                        stop = global_envs_incomplete_tasks[-1] + 1 - num_structs_in_kernel
                    data_files = file_utils.find(test_structs_dir, '*.json')
                    rank_print(rank_output_path, 'About to get atoms list')
                    al = get_atoms_list(data_files[start : stop], rank_output_path)
                    rank_print(rank_output_path, 'Got atoms list')
                    sl = structurelist()
                    sl.count = start + num_structs_in_kernel # Set to the global starting index
                    just_env_start_time = time.time()
                    rank_print(rank_output_path, 'Using parameters', 'c = {}, cotw = {}, n = {}, l = {}, g = {}, cw = {}, zeta = {}, nocenter = {}, exclude = {}, unsoap = {}'.format(c, params.soap_standalone_options['cotw'], n, l, g, params.soap_standalone_options['cw'], zeta, params.soap_standalone_options['nocenter'], params.soap_standalone_options['exclude'], params.soap_standalone_options['unsoap']))
                    for at in al:
                        si = structure(alchem)
                        si.parse(at, c, params.soap_standalone_options['cotw'], n, l, g, params.soap_standalone_options['cw'], params.soap_standalone_options['nocenter'], params.soap_standalone_options['exclude'], unsoap=params.soap_standalone_options['unsoap'], kit=params.soap_standalone_options['kit'])
                        sl.append(si.atomic_envs_matrix, num_atoms_arr, params.user_num_atoms_arr, store_tiled=False)
                    rank_print(rank_output_path, 'time to compute envs for {} structures: {}'.format(stop - start, time.time() - just_env_start_time))
                    del global_envs_incomplete_tasks
                    del si
                    del sl
                    del al
                    rank_print(rank_output_path,'time to read input structures and calculate environments', time.time() - start_time_test_envs)
                    root_print(comm_world.rank,'time to read input structures and calculate environments', time.time() - start_time_test_envs)
                start_time_rect_kernel = time.time()
                # Calculate rectangular matrix if not already calculated
                # Not supporting restarts of this matrix b/c it is so much slower to write one similarity at a time and also requires a comm.barrier.
                # However, we should check if this matrix has been completed which we'll know if the root rank made a file called completed_rect_kernel
                if comm.rank == 0:
                    rect_kernel_memmap_path = os.path.join(current_krr_test_dir, 'rect_kernel.dat')
                    fp = np.memmap(rect_kernel_memmap_path, dtype='float32', mode='w+', shape=(num_test_structs, num_structs_in_kernel))
                    del fp
                    
                    num_unique_species_arr_for_test_structs = write_num_atoms_arr_and_num_unique_species_arr(current_krr_test_dir, data_files)
                    num_unique_species_arr_for_kernel = file_utils.safe_np_load(os.path.join(kernel_calculation_path, 'num_unique_species_arr.npy'), time_frame=0.001, verbose=False, check_file_done_being_written_to=False)
                    num_unique_species_arr = np.hstack((num_unique_species_arr_for_kernel, num_unique_species_arr_for_test_structs))
                else:
                    kernel_tasks = None
                num_atoms_arr_for_kernel = file_utils.safe_np_load(os.path.join(kernel_calculation_path, 'num_atoms_arr.npy'), time_frame=0.001, verbose=False, check_file_done_being_written_to=False)
                num_atoms_arr_for_test_structs = file_utils.safe_np_load(os.path.join(current_krr_test_dir, 'num_atoms_arr.npy'), time_frame=0.001, verbose=False, check_file_done_being_written_to=True)
                num_atoms_arr = np.hstack((num_atoms_arr_for_kernel, num_atoms_arr_for_test_structs))
    
                # Barrier to get a better available memory estimate and doesn't hurt too much because we do a comm.scatter afterwards
                comm.barrier()
                if comm.rank == 0:
                    get_rank_tasks_start_time = time.time()
                    num_ranks_for_nodes = get_num_ranks_for_kernel_computation(overall_krr_test_task_list, n, l, num_atoms_arr, num_unique_species_arr, rank_hostnames, params.max_mem, params.node_mem)
                    rank_print(rank_output_path, 'num_ranks_for_nodes time', time.time() - get_rank_tasks_start_time)    
                else:
                    num_ranks_for_nodes  = None
                
                scatter_start_time = time.time()
                kernel_tasks = comm.scatter(num_ranks_for_nodes, root=0)
                rank_print(rank_output_path, 'kernel_tasks scatter time', time.time() - scatter_start_time)
    
                rank_print(rank_output_path, 'For kernel computation, got {} kernel_tasks'.format(len(kernel_tasks))) #, kernel_tasks)
                loaded_soaps = {}
                # kernel_tasks should be traversal order of the kernel matrix tasks such that it should be a np.array with shape (num_tasks_per_rank, 2) where
                # each row has (as the first element) the struct index of a structure which should get its atomic env matrix repeated and (as the second element)
                # the struct index of a structure which should get its atomic env matrix tiled.
                # kernel_tasks is np.array with shape usually being (num_tasks_per_rank, 2) where each row is a pair of structures that need their similarities evaluated.
                breathing_room_factor = ranks_per_node * breathing_room_factor_per_node
                just_rect_kernel_start_time = time.time()
                my_kernel_data = []
                root_print(comm.rank, 'Beginning rectangular kernel calculation')
                rank_print(rank_output_path, 'Beginning rectangular kernel calculation')
                for task_idx in range(len(kernel_tasks)):
                    if params.verbose:
                        rank_print(rank_output_path, 'computing task_idx for rect kernel', task_idx)
                    num_atoms_in_i, num_atoms_in_j, matrices_to_keep = load_soaps(task_idx, loaded_soaps, kernel_tasks, num_atoms_arr, global_envs_dir, params.lowmem, params.lowestmem, breathing_room_factor, rank_output_path, kernel_calculation_path, krrtest=True)
                    i,j = kernel_tasks[task_idx]
                    if params.verbose:
                        rank_print(rank_output_path, 'i,j', i,j)
                    sij = np.dot(loaded_soaps[matrices_to_keep[0]], loaded_soaps[matrices_to_keep[1]]) ** zeta
                    if params.verbose:
                        rank_print(rank_output_path, 'sij', sij)
                    my_kernel_data.append(sij)
                rank_print(rank_output_path, 'time to compute rectangular kernel entries for {} entries: {}'.format(len(kernel_tasks), time.time() - just_rect_kernel_start_time))
                write_kernel_start_time = time.time()
                kernel_data = comm.gather(my_kernel_data, root=0)
                rank_print(rank_output_path,'time to gather rect kernel_data', time.time() - write_kernel_start_time)
                if comm.rank == 0:
                    kernel_data = list_utils.flatten_list(kernel_data)
                    write_similarities_to_rect_file(num_structs_in_kernel, num_test_structs, kernel_data, rect_kernel_memmap_path, rank_output_path=rank_output_path)
                    rank_print(rank_output_path,'wrote rect kernel_data')
                    
                rank_print(rank_output_path,'ending rect kernel calculation step.')
                end_time_kernel = time.time()
                if comm.rank == 0:
                    rank_print(rank_output_path,'time to calculate rectangular kernel', end_time_kernel - start_time_rect_kernel)
                    root_print(comm_world.rank,'time to calculate rectangular kernel', end_time_kernel - start_time_rect_kernel)
                    rank_print(rank_output_path,'time to gather and write rectangular kernel', end_time_kernel - write_kernel_start_time)
                    file_utils.write_to_file(os.path.join(current_krr_test_dir, 'completed_rect_kernel'), 'completed_rect_kernel', mode='w')
                    del kernel_data
                    del overall_krr_test_task_list
                del my_kernel_data
                del kernel_tasks
                del sij
                # Finally, predict the property values of the test structures.
                if comm.rank == 0:
                    rect_matrix = np.memmap(rect_kernel_memmap_path, dtype='float32', mode='r', shape=(num_test_structs, num_structs_in_kernel))
                    kernel_weights = np.load(kernel_weights_fpath)
                    predicted_atomic_binding_energies = np.dot(rect_matrix, kernel_weights)
                    root_print(comm.rank,'time to get predicted property values', time.time() - krr_test_start_time)
                    rank_print(rank_output_path,'time to get predicted property values', time.time() - krr_test_start_time)
                    np.save(params.test_prop_fname, predicted_atomic_binding_energies)
            else:
                if num_param_combos is None:
                    rank_print(rank_output_path, 'no krr_test; moving on.')
            

    end_time = time.time()
    try:
        rank_print(rank_output_path,'total time', end_time - start_time)
    except:
        pass
    root_print(comm_world.rank,'total time', end_time - start_time)
                        


if __name__ == '__main__':
    params = SetUpParams()
    params.setup_krr_params()
    params.setup_krr_test_params()
    #print(params.soap_param_list)
    soap_workflow(params)
    #print(params.krr_param_list)
