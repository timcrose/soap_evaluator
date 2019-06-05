import os, sys, random, time, subprocess, glob, itertools
import instruct

class SetUpParams():
    def __init__(self):
        '''
        Interprets the instruction and calls the respective attributes of inst
        '''
        inst_path = sys.argv[-1]
        inst = instruct.Instruct()
        inst.load_instruction_from_file(inst_path)
        self.inst = inst
        
        sname = 'train_kernel'
        self.soap_param_list = ['time', 'python']
        self.soap_param_list += [inst.get(sname, 'glosim_path')]

        glosim_soap_options = [['filename', ''],
                                      ['periodic', '--periodic'],
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
                                      ['nonorm','--nonorm'],
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
                                      ['restart','--restart']
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
                                      ['f', '-f'],
                                      ['truetest', '--truetest'],
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
        for option, option_string in [['kernel', ''],
                                      ['weights', ''],
                                      ['props', ''],
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
        elif sname == 'krr_test':
            search_str = 'rect'

        file_list = glob.glob('*')
        for fname in file_list:
            if search_str in fname:
                return fname

    def add_to_param_list(self, inst, sname, option, option_string):
        if 'kernel' in option and 'krr' in sname:
            value = self.get_kernel_fname(sname)
            if sname == 'krr':
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

def modify_soap_hyperparam_list(param_list, params_to_get, params_set):
    '''
    param_list: str
        List of params to pass to glosim to compute kernel.
        Ex:
sys.path.append(os.path.join(os.environ["HOME"], "python_utils")))
         'train.xyz', '--periodic', '-n', '[5, 8]', '-l', '[1]',
         '-c', '[1]', '-g', '[0.7, 1.1]', '--kernel', 'average',
         '--nonorm', '--np', '1']
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
        pos_p = param_list.index('-' + p)
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
    sname = 'cross_val'
    #original working dir
    owd = os.getcwd()

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
                for train_num_structs in inst.get_list(sname, 'train_num_structs'):
                    train_num_structs_path = os.path.join(test_num_structs_path, 'train_num_structs_' + str(train_num_structs))
                    for test_num in range(int(inst.get(sname, 'test_num'))):
                        test_num_path = os.path.join(train_num_structs_path, 'test_num_' + str(test_num))
                        for train_num in range(int(inst.get(sname, 'train_num'))):
                            train_num_path = os.path.join(test_num_path, 'train_num_' + str(train_num))
     
                            if not check_for_and_write_lockfile(train_num_path):
                                continue
                            os.chdir(train_num_path)
                        
                            #Create kernel
                            outfile_path = inst.get('train_kernel', 'outfile')
                            with open(outfile_path, 'w') as f:
                                params.soap_param_list = modify_soap_hyperparam_list(params.soap_param_list,
                                      params_to_get, params_set)

                                step_1 = subprocess.Popen(params.soap_param_list, stdout=f, stderr=f)
                                out_1, err_1 = step_1.communicate()
                        
                            '''
                            #krr
                            params.setup_krr_params()
                            outfile_path = inst.get('krr', 'outfile')
                            with open(outfile_path, 'w') as f:
                                step_2 = subprocess.Popen(params.krr_param_list, stdout=f, stderr=f)
                                out_2, err_2 = step_2.communicate()
                        
                            #create rect file
                            outfile_path = inst.get('create_rect', 'outfile')
                            with open(outfile_path, 'w') as f:
                                params.create_rect_param_list = modify_soap_hyperparam_list(params.create_rect_param_list,
                                      params_to_get, params_set)

                                step_3 = subprocess.Popen(params.create_rect_param_list, stdout=f, stderr=f)
                                out_3, err_3 = step_3.communicate()
                        
                            #perform krr test
                            params.setup_krr_test_params()
                            outfile_path = inst.get('krr_test', 'outfile')
                            with open(outfile_path, 'w') as f:
                                step_4 = subprocess.Popen(params.krr_test_param_list, stdout=f, stderr=f)
                                out_4, err_4 = step_4.communicate()
                            '''


if __name__ == '__main__':
    params = SetUpParams()
    #print(params.soap_param_list)
    soap_workflow(params)
    #print(params.krr_param_list)
