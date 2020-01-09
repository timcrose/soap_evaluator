import os, sys, random, time, subprocess, glob, itertools, json
print('os.environ[LD_LIBRARY_PATH] in batch soap', os.environ['LD_LIBRARY_PATH'])
if '/home/trose/lib64' not in os.environ['LD_LIBRARY_PATH']:
    os.environ['LD_LIBRARY_PATH'] = '/home/trose/lib64:/opt/ohpc/pub/intel/intel18/compilers_and_libraries_2018.0.128/linux/mpi/intel64/lib:/opt/ohpc/pub/intel/intel18/compilers_and_libraries_2018.0.128/linux/mpi/mic/lib:/opt/ohpc/pub/compiler/gcc/5.4.0/lib64'
print('os.environ[LD_LIBRARY_PATH] in batch soap 2nd time', os.environ['LD_LIBRARY_PATH'])
print('os.environ[PYTHONPATH] in batch soap', os.environ['PYTHONPATH'])
print('sys.path', sys.path)
print('listdir', os.listdir('/home/trose/lib64'))
print('liblapack in listdir', 'liblapack.so.3' in os.listdir('/home/trose/lib64'))
print('echoing')
os.system('echo $LD_LIBRARY_PATH')
import numpy as np
import instruct
from copy import deepcopy
from scipy.spatial.distance import pdist
sys.path.append(os.environ["HOME"])
from python_utils import file_utils
print('about to import quippy in batch soap')
import quippy
print('imported quippy in batch soap')
from ibslib.io import read
from libmatch.environments import alchemy, environ
from libmatch.structures import structure, structurelist
from tools import krr
import psutil
from mpi4py import MPI
comm = MPI.COMM_WORLD
MPI_ANY_SOURCE = MPI.ANY_SOURCE

def get_atoms_list(data_files):
    '''
    data_files: list of str
        List of paths to .json files, each with a structure.

    Return: atoms_list
    atoms_list: list of quippy Atoms objects
        List of quippy Atoms objects.

    Purpose: Read each data_file in data_files into an ase atoms object and
        then convert to quippy Atoms object.
    '''
    rank_print(comm.rank, 'num data files', len(data_files))
    rank_print(comm.rank, 'converting jsons to quippy atoms objects')
    atoms_list = [quippy.convert.ase_to_quip(read(data_file).get_ase_atoms()) for data_file in data_files]
    rank_print(comm.rank, 'Converted jsons to quippy atoms objects')
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

        file_list = glob.glob(os.path.join(start_path, '*' + test_str))
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
    zipped_incomplete_tasks = zip(incomplete_tasks_rows, incomplete_tasks_cols)
    # sort because element i,j in the matrix is the same as j,i so we will remove the j,i
    sorted_incomplete_tasks = np.sort(zipped_incomplete_tasks)
    if sorted_incomplete_tasks.size == 0:
        return np.array([])
    # remove the j,i. Now we only have one element in incomplete tasks for every true task.
    incomplete_tasks = np.unique(sorted_incomplete_tasks, axis=0)
    return incomplete_tasks

def progress_being_made(kernel_outfile_fpath):
        '''
        kernel_outfile_fpath: str
            file path of the glosim output when generating the kernel.

        Return: bool
            True: calculation of the kernel is ongoing
            False: o/w
        '''
        kernel_outfile_fsize = os.path.getsize(kernel_outfile_fpath)
        time.sleep(200)
        return kernel_outfile_fsize < os.path.getsize(kernel_outfile_fpath) or kernel_done_calculating(kernel_outfile_fpath)


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


def get_krr_task_list(sname, param_path):
    ntrain_paths = set(file_utils.find(param_path, 'ntrain_*', recursive=True))
    # As a quick approximation, assume that if the output file "saved" exists, then this ntrain_path is done. TODO look
    # inside the files to see for sure if all of the ntests were completed to be more sure if this ntrain_path is done or not.
    saved_paths = set(file_utils.find(param_path, 'saved', recursive=True))
    krr_task_list = np.array(list(ntrain_paths - saved_paths))
    return krr_task_list


def get_specific_krr_params(ntrain_path):
    # modify krr param list for this specific task. The parameters can be extracted from ntrain_path
    ntrain = int(os.path.basename(ntrain_path).split('_')[-1])
    ntest = int(ntrain_path.split('/')[-2].split('_')[-1])
    selection_method = ntrain_path.split('/')[-3]
    return ntrain, ntest, selection_method


def root_print(comm, *print_message):
    if comm.rank == 0:
        print(print_message)

def rank_print(rank, *print_message):
    with open('output_' + str(rank) + '.out', mode='a') as f:
        print(print_message, file=f)

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


def get_traversal_order(incomplete_tasks, kernel_calculation_path):
    '''
    incomplete_tasks: np.array shape (num incomplete tasks, 2)
        Each row is a pair of structure indices that need their similarities evaluated.

    kernel_calculation_path: str
        Path where the kernel.dat and num_atoms_arr.npy are located

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
    num_atoms_arr = np.load(os.path.join(kernel_calculation_path, 'num_atoms_arr.npy'))
    if np.all(num_atoms_arr == num_atoms_arr[0]):
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
        for loaded_soap in loaded_soaps:
            if loaded_soap not in matrices_to_keep:
                del loaded_soaps[loaded_soap]
                break


def load_soaps(task_idx, loaded_soaps, task_list, num_atoms_arr, global_envs_dir, lowmem, breathing_room_factor):
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

        If lowmem is False, load matrices needed for the current calculation but do not delete matrices that were loaded already.
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
    one_matrix = loaded_soaps[loaded_soaps.keys()[0]]
    bytes_per_atom = one_matrix.nbytes / one_matrix.shape[0]
    del one_matrix
    float32_size = 4
    num_atoms_in_i = num_atoms_arr[i]
    #print('num_atoms_in_i', num_atoms_in_i)
    num_atoms_in_j = num_atoms_arr[j]

    matrices_to_keep = [str(i), str(j), str(i) + '_repeat_' + str(num_atoms_in_j), str(j) + '_tile_' + str(num_atoms_in_i)]

    if str(i) + '_repeat_' + str(num_atoms_in_j) not in loaded_soaps:    
        if str(i) not in loaded_soaps:
            if lowmem:
                delete_some_unnecessary_matrices(loaded_soaps, matrices_to_keep, num_atoms_in_i * bytes_per_atom * breathing_room_factor)
            loaded_soaps[str(i)] = np.load(os.path.join(global_envs_dir, 'atomic_envs_matrix_' + str(i) + '.npy'))
        if lowmem:
            delete_some_unnecessary_matrices(loaded_soaps, matrices_to_keep, num_atoms_in_i * num_atoms_in_j * bytes_per_atom * breathing_room_factor)
        #print('loaded_soaps['+str(i)+']', loaded_soaps[str(i)])
        loaded_soaps[str(i) + '_repeat_' + str(num_atoms_in_j)] = np.repeat(loaded_soaps[str(i)], num_atoms_in_j, axis=0).flatten()

    #Could insert a statement here to delete loaded_soaps[str(i)] if it's not needed in any future calculations to further save on memory

    if str(j) + '_tile_' + str(num_atoms_in_i) not in loaded_soaps:
        if str(j) not in loaded_soaps:
            if lowmem:
                delete_some_unnecessary_matrices(loaded_soaps, matrices_to_keep, num_atoms_in_j * bytes_per_atom * breathing_room_factor)
            loaded_soaps[str(j)] = np.load(os.path.join(global_envs_dir, 'atomic_envs_matrix_' + str(j) + '.npy'))
        if lowmem:
            delete_some_unnecessary_matrices(loaded_soaps, matrices_to_keep, num_atoms_in_i * num_atoms_in_j * bytes_per_atom * breathing_room_factor)
        loaded_soaps[str(j) + '_tile_' + str(num_atoms_in_i)] = np.tile(loaded_soaps[str(j)], (num_atoms_in_i, 1)).flatten()

    #Could insert a statement here to delete loaded_soaps[str(j)] if it's not needed in any future calculations to further save on memory
    #Could actually see if a particular matrix will be used later on and prefer to keep it if you could instead delete one that will not be used later on
    #Could try writing and reading the repeated and tiled matrices to and from a file but this might be slower.
    return num_atoms_in_i, num_atoms_in_j


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
    sname = 'krr'
    #original working dir
    owd = os.getcwd()
    soap_runs_dir = os.path.join(owd, 'soap_runs')
    print('soap workflow', flush=True)
    params_to_get = ['n','l','c','g']
    params_list = [inst.get_list('calculate_kernel', p) for p in params_to_get]
    params_combined_iterable = itertools.product(*params_list)

    root_print(comm, 'using ' + str(comm.size) + ' MPI ranks on ' + str(params.num_structures_to_use) + ' structures.')
    start_time = time.time()

    for params_set in params_combined_iterable:
        start_time_param = time.time()
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
        n, l, c, g = params_set
        
        param_path = os.path.join(soap_runs_dir, param_string)
        print('param_path', param_path, flush=True)
        kernel_calculation_path = os.path.abspath(os.path.join(param_path, 'calculate_kernel'))
        params.kernel_calculation_path = kernel_calculation_path
        en_all_dat_fname = inst.get('krr', 'props')
        en_all_dat_fpath = os.path.join(kernel_calculation_path, en_all_dat_fname)
        
        # Calculate global environments if not already calculated
        global_envs_dir = os.path.join(kernel_calculation_path, 'tmpstructures')
        load_balancing_factor = 1.0 # Range > 0 and <= 1 where high means all computations are expected to take the same amount of time and
        #                             low means certain samples will take longer than others. Setting less than 1 in this case helps with 
        #                             load balancing but there's more communication.
        start_time_envs = time.time()
        xyz_fpath = os.path.join(kernel_calculation_path, params.xyz_file_basename + '.xyz')
        if comm.rank == 0:
            num_structures = len(file_utils.get_lines_of_file(en_all_dat_fpath))
        else:
            num_structures = None
        print('about to bcast', flush=True)
        num_structures = comm.bcast(num_structures, root=0)
        root_print(comm, str(num_structures) + ' structures are in the pool.')
        # make an array that will store the number of atoms in every structure.
        # Must have at least 2 ranks so that 1 can be the master rank
        num_tasks_per_rank = int(np.ceil((load_balancing_factor * num_structures) / (comm.size - 1)))
        if comm.rank == 0:
            file_utils.mkdir_if_DNE(global_envs_dir)
            if os.path.exists(global_envs_dir):
                global_envs_incomplete_tasks = np.array(list(set(np.arange(num_structures)) - set([int(file_utils.fname_from_fpath(fpath).split('_')[1]) for fpath in file_utils.find(global_envs_dir, '*.dat')])))
            else:
                global_envs_incomplete_tasks = np.arange(num_structures)
            done_ranks = []
            num_atoms_lst = []
            while len(done_ranks) != comm.size - 1:
                # world rank 0 waiting for ready rank
                ready_rank = comm.recv(source=MPI_ANY_SOURCE, tag=0)

                # a rank is now ready to recieve a message
                if len(global_envs_incomplete_tasks) != 0:
                    message = global_envs_incomplete_tasks[:num_tasks_per_rank]
                    global_envs_incomplete_tasks = np.delete(global_envs_incomplete_tasks, np.arange(num_tasks_per_rank))
                else: #Done!
                    message = 'done'
                    done_ranks.append(ready_rank)
                # world rank 0 sending message to ready_rank
                print('rank', ready_rank, 'is ready to calculate environments for structures indexed by', message)
                comm.send(message, dest=ready_rank, tag=1)
                # world rank 0 sent message to ready_rank
        else:
            alchem = alchemy(mu=params.soap_standalone_options['mu'])
            os.chdir(kernel_calculation_path)
            data_files = file_utils.glob(os.path.join(params.structure_dir, '*.json'))
            if params.num_structures_to_use != 'all':
                data_files = data_files[: int(params.num_structures_to_use)]
            
            while True:
                comm.send(comm.rank, dest=0, tag=0)
                # ready_rank told comm.rank 0 that it is ready to recieve a message from comm.rank 0
                message = comm.recv(source=0, tag=1)
                # read_rank received message from comm.rank 0
                if (type(message) is str and message == 'done') or len(message) == 0:
                    break
                # message is a np.array of tasks where each task is the index in the structure list of a structure that still
                # needs its global soap descriptor calculated.
                # Calculate the global soap descriptor for each structure whose index is in message
                start = message[0]
                if message[-1] == start:
                    stop = start + 1
                else:
                    stop = message[-1] + 1
                rank_print(comm.rank, 'about to get atoms list')
                al = get_atoms_list(data_files[start : stop])
                rank_print(comm.rank, 'got atoms list')
                sl = structurelist()
                sl.count = start # Set to the global starting index
                for at in al:
                    si = structure(alchem)
                    si.parse(at, c, params.soap_standalone_options['cotw'], n, l, g, params.soap_standalone_options['cw'], params.soap_standalone_options['nocenter'], params.soap_standalone_options['exclude'], unsoap=params.soap_standalone_options['unsoap'], kit=params.soap_standalone_options['kit'])
                    sl.append(si.atomic_envs_matrix)
        start_time_kernel = time.time()
        root_print(comm,'time to read input structures and calculate environments', start_time_kernel - start_time_envs)
        # Calculate kernel if not already calculated
        kernel_memmap_path = os.path.join(kernel_calculation_path, 'kernel.dat')
        if comm.rank == 0:
            if os.path.exists(kernel_memmap_path):
                incomplete_tasks = get_incomplete_tasks(kernel_memmap_path, (num_structures, num_structures))
            else:
                incomplete_tasks_rows, incomplete_tasks_cols = np.triu_indices(num_structures)
                incomplete_tasks = np.array(zip(incomplete_tasks_rows, incomplete_tasks_cols))
                fp = np.memmap(kernel_memmap_path, dtype='float32', mode='w+', shape=(num_structures, num_structures))
                # Initialize to -11111.0 just so we know that if it's -11111.0 that it's incomplete
                fp[:] = np.ones((num_structures, num_structures), dtype='float32') * -11111.0
                del fp
            print('getting kernel traversal order')
            incomplete_tasks = get_traversal_order(incomplete_tasks, kernel_calculation_path)
            done_ranks = []
            while len(done_ranks) != comm.size - 1:
                # world rank 0 waiting for ready rank
                ready_rank = comm.recv(source=MPI_ANY_SOURCE, tag=0)
                # a rank is now ready to recieve a message
                if len(incomplete_tasks) != 0:
                    message = incomplete_tasks[:num_tasks_per_rank]
                    incomplete_tasks = np.delete(incomplete_tasks, np.arange(num_tasks_per_rank), axis=0)
                else: #Done!
                    message = 'done'
                    done_ranks.append(ready_rank)
                # world rank 0 sending message to ready_rank
                comm.send(message, dest=ready_rank, tag=1)
                # world rank 0 sent message to ready_rank
        else:
            loaded_soaps = {}
            num_atoms_arr = np.load(os.path.join(kernel_calculation_path, 'num_atoms_arr.npy'))
            while True:
                comm.send(comm.rank, dest=0, tag=0)
                # ready_rank told comm.rank 0 that it is ready to recieve a message from comm.rank 0
                message = comm.recv(source=0, tag=1)
                # read_rank received message from comm.rank 0
                if (type(message) is str and message == 'done') or len(message) == 0:
                    break

                # message should be traversal order of the kernel matrix tasks such that it should be a np.array with shape (num_tasks_per_rank, 2) where
                # each row has (as the first element) the struct index of a structure which should get its atomic env matrix repeated and (as the second element)
                # the struct index of a structure which should get its atomic env matrix tiled.
                # message is np.array with shape usually being (num_tasks_per_rank, 2) where each row is a pair of structures that need their similarities evaluated.
                breathing_room_factor = min([12.0, comm.size])
                for task_idx in range(len(message)):
                    num_atoms_in_i, num_atoms_in_j = load_soaps(task_idx, loaded_soaps, message, num_atoms_arr, global_envs_dir, params.lowmem, breathing_room_factor)
                    i,j = message[task_idx]
                    sij = np.dot(loaded_soaps[str(i) + '_repeat_' + str(num_atoms_in_j)], loaded_soaps[str(j) + '_tile_' + str(num_atoms_in_i)])
                    write_similarities_to_file(i, j, sij, num_structures, kernel_memmap_path)
                    write_similarities_to_file(j, i, sij, num_structures, kernel_memmap_path)
        start_time_krr = time.time()
        root_print(comm,'time to calculate kernel', start_time_krr - start_time_kernel)
        # krr. Currently haven't implemented krr_test into the mpi version.
        if 'krr' not in params.process_list:
            return
        if comm.rank == 0:
            krr_task_list = get_krr_task_list(sname, param_path)
            done_ranks = []
            while len(done_ranks) != comm.size - 1:
                # world rank 0 waiting for ready rank
                ready_rank = comm.recv(source=MPI_ANY_SOURCE, tag=0)
                # a rank is now ready to recieve a message
                if len(krr_task_list) != 0:
                    message = krr_task_list[:num_tasks_per_rank]
                    krr_task_list = np.delete(krr_task_list, np.arange(num_tasks_per_rank))
                else: #Done!
                    message = 'done'
                    done_ranks.append(ready_rank)
                # world rank 0 sending message to ready_rank
                comm.send(message, dest=ready_rank, tag=1)
                # world rank 0 sent message to ready_rank
        else:
            params.setup_krr_params()
            outfile_path = inst.get(sname, 'outfile')
            props_fname = params.krr_param_list[params.krr_param_list.index('--props') + 1]
            props_path = os.path.join(kernel_calculation_path, os.path.basename(os.path.abspath(props_fname)))
            sys_stdout_original = sys.stdout
            while True:
                comm.send(comm.rank, dest=0, tag=0)
                # ready_rank told comm.rank 0 that it is ready to recieve a message from comm.rank 0
                message = comm.recv(source=0, tag=1)
                # read_rank received message from comm.rank 0
                if (type(message) is str and message == 'done') or len(message) == 0:
                    break
                # message is np.array with shape (,number of tasks) where each element is an ntrain path.
                
                for ntrain_path in message:
                    os.chdir(ntrain_path)
                    ntrain, ntest, selection_method = get_specific_krr_params(ntrain_path)
                    # do krr
                    sys.stdout = open(outfile_path, 'w')
                    krr.main(kernels=[kernel_memmap_path], props=[props_path], kweights=params.krr_standalone_options['kweights'], mode=selection_method, ntrain=ntrain,
                                    ntest=ntest, ntrue=params.krr_standalone_options['ntrue'], csi=params.krr_standalone_options['csi'],
                                    sigma=params.krr_standalone_options['sigma'], ntests=params.krr_standalone_options['ntests'], 
                                    savevector=params.krr_standalone_options['saveweights'], refindex=params.krr_standalone_options['refindex'],
                                    inweights=params.krr_standalone_options['pweights'])
        end_time_krr = time.time()
        root_print(comm,'time for krr', end_time_krr - start_time_krr)
        root_print(comm,'time for param', end_time_krr - start_time_param)
    end_time = time.time()
    root_print(comm,'total time', end_time - start_time)
                        


if __name__ == '__main__':
    params = SetUpParams()
    #print(params.soap_param_list)
    soap_workflow(params)
    #print(params.krr_param_list)
