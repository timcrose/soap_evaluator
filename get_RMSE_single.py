from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import instruct, sys, os
sys.path.append('/home/trose/python_utils')
import file_utils2 as file_utils

def get_ref_and_pred_energies_from_test_results(test_results_fname):
    if not os.path.isfile(test_results_fname):
        return np.array([]), np.array([])
    with open(test_results_fname) as f:
        lines = f.readlines()
    line = lines[0].split()
    ref_energies = np.array([])
    pred_energies = np.array([])
    i = 0
    while line[0] == str(i):
        ref_energies = np.append(ref_energies, [float(line[1])])
        pred_energies = np.append(pred_energies, [float(line[2])])
        i += 1
        line = lines[i].split()
    return ref_energies, pred_energies

def main():
    inst_path = sys.argv[-1]
    inst = instruct.Instruct()
    inst.load_instruction_from_file(inst_path)

    owd = os.getcwd()

    sname = 'cross_val'
    for selection_method in inst.get_list(sname, 'selection_methods'):
        selection_method_path = os.path.join(owd, selection_method)

        params_to_get = ['n','l','c','g']
        param_string = ''
        for p in params_to_get:
            param_string += p
            param_to_add = inst.get('train_kernel', p)
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

                #For each train num_structs, combine the RMSE's of each replica and print this to 
                # a file along with test_num_structs and train_num_structs.
                ref_energies = np.array([])
                pred_energies = np.array([])

                for test_num in range(int(inst.get(sname, 'test_num'))):
                    test_num_path = os.path.join(train_num_structs_path, 'test_num_' + str(test_num))
                    for train_num in range(int(inst.get(sname, 'train_num'))):
                        train_num_path = os.path.join(test_num_path, 'train_num_' + str(train_num))

                        test_results_path = os.path.join(train_num_path, 'test_results')
                        ref_energies_single_file, pred_energies_single_file = \
                                   get_ref_and_pred_energies_from_test_results(test_results_path)

                        ref_energies = np.append(ref_energies, ref_energies_single_file)
                        pred_energies = np.append(pred_energies, pred_energies_single_file)
                print(test_num_structs, train_num_structs)
                print(len(pred_energies))
                print(ref_energies)
                if len(pred_energies) != 0:
                    rmse = sqrt(mean_squared_error(ref_energies, pred_energies))
                    file_utils.write_row_to_csv('learning_curve_' + param_string + '_test_num_structs_' + str(test_num_structs) + '.csv', [train_num_structs, rmse])
main()
