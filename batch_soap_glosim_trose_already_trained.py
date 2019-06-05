import os, sys, random, time, subprocess, glob, itertools
import instruct
from filelock import FileLock
sys.path.append(os.path.join(os.environ["HOME"], "python_utils"))
import file_utils

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
        self.kernel_lockfile = 'kernel_lockfile'

        glosim_soap_options = [['filename', ''],
                                      ['separate_species', '--separate_species'],
                                      ['exclude','--exclude'],
                                      ['nocenter','--nocenter'],
                                      ['envsim','--envsim'],
                                      ['verbose','--verbose'],
                                      ['n','-n'],
                                      ['l','-l'],
                                      ['c','-c'],
                                      ['cotw','--cotw'],
                                      ['g','-g'],
                                      ['cw','-cw'],
                                      ['mu','--mu'],
                                      ['usekit','--usekit'],
                                      ['gamma','--gamma'],
                                      ['zeta','--zeta'],
                                      ['kit','--kit'],
                                      ['alchemy_rules','--alchemy_rules'],
                                      ['kernel','--kernel'],
                                      ['peratom', '--peratom'],
                                      ['unsoap', '--unsoap'],
                                      ['normalize_global', '--normalize_global'],
                                      ['onpy', '--onpy'],
                                      ['permanenteps','--permanenteps'],
                                      ['distance','--distance'],
                                      ['np','--np'],
                                      ['ij','--ij'],
                                      ['nlandmarks','--nlandmarks'],
                                      ['first','--first'],
                                      ['last','--last'],
                                      ['reffirst','--reffirst'],
                                      ['reflast','--reflast'],
                                      ['refxyz','--refxyz'],
                                      ['prefix','--prefix'],
                                      ['livek','--livek'],
                                      ['lowmem','--lowmem'],
                                      ['restart','--restart'],
                                     ]

        for option, option_string in glosim_soap_options:
            param_to_add = self.add_to_param_list(inst, sname, option, option_string)
            if param_to_add is not None:
                self.soap_param_list += param_to_add

        sname = 'create_rect'
        self.create_rect_param_list = ['time', 'python']
        self.create_rect_param_list += [inst.get(sname, 'glosim_path')]
        for option, option_string in glosim_soap_options:
            param_to_add = self.add_to_param_list(inst, sname, option, option_string)
            if param_to_add is not None:
                self.create_rect_param_list += param_to_add

    def setup_krr_params(self):
        sname = 'krr'
        self.krr_param_list = ['time', 'python']
        self.krr_param_list += [self.inst.get(sname, 'krr_path')]
        for option, option_string in [['kernels', '--kernels'],
                                      ['props', '--props'],
                                      ['kweights', '--kweights'],
                                      ['mode', '--mode'],
                                      ['ntrain', '--ntrain'],
                                      ['ntest', '--ntest'],
                                      ['ntrue', '--ntrue'],
                                      ['csi', '--csi'],
                                      ['sigma', '--sigma'],
                                      ['ntests', '--ntests'],
                                      ['pweights', '--pweights'],
                                      ['refindex', '--refindex'],
                                      ['saveweights', '--saveweights']
                                     ]:

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
        elif sname == 'krr_test':
            search_str = 'rect'
            search_str_avoid = 'no_strs_to_avoid'

        file_list = glob.glob(os.path.join(self.kernel_calculation_path, '*'))
        for fname in file_list:
            if search_str in fname and search_str_avoid not in fname:
                return fname
        return None


    def add_to_param_list(self, inst, sname, option, option_string):
        if 'kernel' in option and 'krr' in sname:
            value = self.get_kernel_fname(sname)
            if sname == 'krr' or sname == 'krr_test':
                return [option_string, value]
            return [value]
            
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


def kernel_done_calculating(kernel_outfile_fpath):
    '''
    kernel_outfile_fpath: str
        file path of the glosim output when generating the kernel.

    Return: bool
        True: kernel computation has completed
        False: o/w

    Purpose: processes that are not part of the kernel generation process are waiting for the kernel generation to be
        finished. It will output "Success" in that file when complete.
    '''
    return file_utils.grep('uccess', kernel_outfile_fpath, read_mode='r', return_list=False, search_from_top_to_bottom=False)


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

        kernel_calculation_path = os.path.abspath(os.path.join(param_path, 'calculate_kernel'))

        params.kernel_calculation_path = kernel_calculation_path
        outfile_fname = inst.get('calculate_kernel', 'outfile')
        outfile_fpath = os.path.join(kernel_calculation_path, outfile_fname)
        while not kernel_done_calculating(outfile_fpath):
            # Attempt to acquire lockfile.
            time.sleep(random.random())
            with FileLock(params.kernel_lockfile, dir_name=kernel_calculation_path,timeout=10, delay=1, exit_gracefully=True) as fl:
                if not fl.timed_out and not kernel_done_calculating(outfile_fpath):
                    #Create kernel
                    params.soap_param_list = modify_soap_hyperparam_list(params.soap_param_list,
                                params_to_get, params_set)
                    
                    os.chdir(kernel_calculation_path)
                    
                    with open(outfile_fpath, 'w') as f:
                        step_1 = subprocess.Popen(params.soap_param_list, stdout=f, stderr=f)
                        out_1, err_1 = step_1.communicate()

            if kernel_done_calculating(outfile_fpath):
                break
            elif not progress_being_made(kernel_calculation_path):
                print('returning because progress on kernel generation not being made.')
                return

        for selection_method in inst.get_list(sname, 'mode'):
            selection_method_path = os.path.join(param_path, selection_method)
            
            for ntest in inst.get_list(sname, 'ntest'):
                ntest_path = os.path.join(selection_method_path, 'ntest_' + str(ntest))
                for ntrain in inst.get_list(sname, 'ntrain'):
                    ntrain_path = os.path.abspath(os.path.join(ntest_path, 'ntrain_' + str(ntrain)))

                    time.sleep(random.random())

                    if not check_for_and_write_lockfile(ntrain_path):
                        continue
                    os.chdir(ntrain_path)
                
                    if 'krr' in params.process_list:
                        #krr
                        params.setup_krr_params()
                        outfile_path = inst.get(sname, 'outfile')
                        props_fname = params.krr_param_list[params.krr_param_list.index('--props') + 1]
                        props_path = os.path.join(kernel_calculation_path, os.path.basename(os.path.abspath(props_fname)))
                        
                        params.krr_param_list = modify_soap_hyperparam_list(params.krr_param_list,
                                    ['ntrain', 'ntest', 'mode', 'props'], 
                                    [ntrain, ntest, selection_method, props_path],
                                     dashes='--')

                        with open(outfile_path, 'w') as f:
                            step_2 = subprocess.Popen(params.krr_param_list, stdout=f, stderr=f)
                            out_2, err_2 = step_2.communicate()

                    if 'krr_test' in params.process_list:
                        params.setup_krr_test_params()
                        outfile_path = inst.get(sname, 'outfile')
                        
                        


if __name__ == '__main__':
    params = SetUpParams()
    #print(params.soap_param_list)
    soap_workflow(params)
    #print(params.krr_param_list)
