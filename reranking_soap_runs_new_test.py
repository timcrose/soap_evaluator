import instruct, itertools, os, sys
sys.path.append(os.path.join(os.environ['HOME'], 'python_utils'))
import file_utils, math_utils


def get_result(search_str, results_list):
    for i, el in enumerate(results_list):
        if search_str in el:
            idx = i
            break

    value = results_list[idx][len(search_str) + 1:].rstrip()
    
    return float(value)


def get_results(inst):
    '''
    inst: contains conf file params

    Purpose: Get results from a test on a new set
    '''
    owd = os.getcwd()
    soap_runs_dir = os.path.join(owd, 'soap_runs')
    sname = 'krr'
    all_results_list = [['ntrain', 'ntest', 'MAE', 'RMSE', 'SUP', 'R2']]

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

        calculate_kernel_path = os.path.join(param_path, 'calculate_kernel')

        all_xyz_structs_fpath = inst.get('calculate_kernel', 'filename')

        for selection_method in inst.get_list(sname, 'mode'):
            selection_method_path = os.path.join(param_path, selection_method)

            for ntest in inst.get_list(sname, 'ntest'):
                ntest_path = os.path.join(selection_method_path, 'ntest_' + str(ntest))

                for ntrain in inst.get_list(sname, 'ntrain'):
                    ntrain_path = os.path.join(ntest_path, 'ntrain_' + str(ntrain))

                    MAE_tmp = []
                    RMSE_tmp = []
                    SUP_tmp = []
                    R2_tmp = []
                    ntests = inst.get_eval(sname, 'ntests')
                    for itest in range(ntests):
                        saved_results_fname = os.path.basename(inst.get('krr_test', 'outfile')) + '_itest' + str(itest)
                        saved_results_fpath = os.path.join(ntrain_path, saved_results_fname)
                        test_results = file_utils.grep('# Test points  MAE', saved_results_fpath)
                        print(saved_results_fpath)
                        test_results = test_results[0].split()
                        
                        MAE_tmp.append(get_result('MAE', test_results))
                        RMSE_tmp.append(get_result('RMSE', test_results))
                        SUP_tmp.append(get_result('SUP', test_results))
                        R2_tmp.append(get_result('R2', test_results))

                    MAE = math_utils.mean(MAE_tmp)
                    RMSE = math_utils.mean(RMSE_tmp)
                    SUP = math_utils.mean(SUP_tmp)
                    R2 = math_utils.mean(R2_tmp)

                    all_results_list.append([ntrain,ntest,MAE, RMSE, SUP, R2])
            file_utils.write_rows_to_csv('soap_results_' + selection_method + '_' + param_string + '_new_test.csv', all_results_list)
            all_results_list = [['ntrain', 'ntest', 'MAE', 'RMSE', 'SUP', 'R2']]


def main():
    '''
    Interprets the instruction and calls the respective attributes of inst
    '''
    owd = os.getcwd()

    inst_path = sys.argv[-1]
    inst = instruct.Instruct()
    inst.load_instruction_from_file(inst_path)

    get_results(inst)

if __name__ == '__main__':
    main()
