import os, sys, random, time, subprocess, glob, itertools
import numpy as np
import instruct
from filelock import FileLock
sys.path.append(os.environ["HOME"])
from python_utils import file_utils
import quippy
from libmatch.environments import alchemy, environ
from libmatch.structures import structk, structure, structurelist
from tools import krr
from mpi4py import MPI
comm = MPI.COMM_WORLD
MPI_ANY_SOURCE = MPI.ANY_SOURCE

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
        self.process_list = inst.get_list(sname, 'sections')
        
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
        values in the kernel is 11111.0

    Purpose: Determine which, if any, similarities have not yet been computed.
    '''
    fp = np.memmap(kernel_memmap_path, dtype='float32', mode='r', shape=shape)
    incomplete_tasks_rows, incomplete_tasks_cols = np.where(fp == 11111.0)
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


def check_for_and_write_lockfile(wdir):
    '''
    wdir: str
        path of the working directory where lockfiles should be placed

    Purpose: Check if there is a lockfile. If so, move on to look in another
        dir. If not, create a lockfile and work there
    '''
    #Sleep random amount to avoid clashes between processes
    time.sleep(random.randint(0,100)/25.0)
    lockfile_path = os.path.join(wdir, 'lockfile')
    if os.path.isfile(lockfile_path):
        return False
    with open(lockfile_path, 'w') as f:
        f.write('locked')
    return True

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


def task_complete(i, j, num_structures, kernel_memmap_path):
    '''
    Return: bool
        True: task is already complete
        False: task is incomplete
    '''
    float32_size = 4
    offset = (i * num_structures + j) * float32_size
    fp = np.memmap(kernel_memmap_path, dtype='float32', mode='r', offset=offset, shape=(1, 1))
    return fp[0][0] != 11111.0


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

    params_to_get = ['n','l','c','g']
    params_list = [inst.get_list('calculate_kernel', p) for p in params_to_get]
    params_combined_iterable = itertools.product(*params_list)
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
        if comm.rank == 0:
            num_structures = len(file_utils.get_lines_of_file(en_all_dat_fpath))
        else:
            num_structures = None
        if comm.rank == 0:
            file_utils.mkdir_if_DNE(global_envs_dir)
            # Must have at least 2 ranks so that 1 can be the master rank
            num_tasks_per_rank = int(np.ceil((load_balancing_factor * num_structures) / (comm.size - 1)))
            if os.path.exists(global_envs_dir):
                global_envs_incomplete_tasks = np.array(list(set(np.arange(num_structures)) - set([int(file_utils.fname_from_fpath(fpath).split('_')[1]) for fpath in file_utils.find(global_envs_dir, '*.dat')])))
            else:
                global_envs_incomplete_tasks = np.arange(num_structures)
            done_ranks = []
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
                comm.send(message, dest=ready_rank, tag=1)
                # world rank 0 sent message to ready_rank
        else:
            alchem = alchemy(mu=params.soap_standalone_options['mu'])
            os.chdir(kernel_calculation_path)
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
                al = quippy.AtomsList(params.soap_standalone_options['filename'], start=start, stop=stop)
                sl = structurelist()
                sl.count = start # Set to the global starting index
                for at in al:
                    si = structure(alchem)
                    si.parse(at, c, params.soap_standalone_options['cotw'], n, l, g, params.soap_standalone_options['cw'], params.soap_standalone_options['nocenter'], params.soap_standalone_options['exclude'], unsoap=params.soap_standalone_options['unsoap'], kit=params.soap_standalone_options['kit'])
                    sl.append(si)
        num_structures = comm.bcast(num_structures, root=0)
        start_time_kernel = time.time()
        root_print(comm,'time to calculate environments', start_time_kernel - start_time_envs)
        # Calculate kernel if not already calculated
        kernel_memmap_path = os.path.join(kernel_calculation_path, 'kernel.dat')
        if comm.rank == 0:
            if os.path.exists(kernel_memmap_path):
                incomplete_tasks = get_incomplete_tasks(kernel_memmap_path, (num_structures, num_structures))
            else:
                incomplete_tasks_rows, incomplete_tasks_cols = np.triu_indices(num_structures)
                incomplete_tasks = np.array(zip(incomplete_tasks_rows, incomplete_tasks_cols))
                fp = np.memmap(kernel_memmap_path, dtype='float32', mode='w+', shape=(num_structures, num_structures))
                # Initialize to 11111.0 just so we know that if it's 11111.0 that it's incomplete
                fp[:] = (np.ones((num_structures, num_structures), dtype='float32') * 11111.0)[:]
                del fp
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
            sl = structurelist()
            while True:
                comm.send(comm.rank, dest=0, tag=0)
                # ready_rank told comm.rank 0 that it is ready to recieve a message from comm.rank 0
                message = comm.recv(source=0, tag=1)
                # read_rank received message from comm.rank 0
                if (type(message) is str and message == 'done') or len(message) == 0:
                    break
                # message is np.array usually shape (num_tasks_per_rank, 2) where each row is a pair of structures that need their similarities evaluated.
                unique_struct_ids = np.unique(message)
                unique_struct_envs = [sl[iframe] for iframe in unique_struct_ids]
                for i, j in message:
                    i_index = np.where(unique_struct_ids == i)[0][0]
                    j_index = np.where(unique_struct_ids == j)[0][0]
                    # Check if task was already completed by another program (instance of batch_soap_glosim_mpi on a separate run)
                    if task_complete(i, j, num_structures, kernel_memmap_path):
                        continue
                    sij,_ = structk(unique_struct_envs[i_index], unique_struct_envs[j_index], alchem, params.soap_standalone_options['peratom'], mode=params.soap_standalone_options['kernel'], fout=None, peps=params.soap_standalone_options['permanenteps'], gamma=params.soap_standalone_options['gamma'], zeta=params.soap_standalone_options['zeta'], xspecies=params.soap_standalone_options['separate_species'])
                    write_similarities_to_file(i, j, sij, num_structures, kernel_memmap_path)
                    write_similarities_to_file(j, i, sij, num_structures, kernel_memmap_path)
        comm.barrier()
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
