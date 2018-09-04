from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import instruct, sys, os
sys.path.append('/home/trose/python_utils')
import file_utils

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

    sname = 'cross_val'
    for selection_method in inst.get_list(sname, 'selection_methods'):
        selection_method_path = os.path.abspath(selection_method)

        param_strings = [os.path.basename(param_path) for param_path in file_utils.glob(os.path.join(selection_method_path, '*'))]
        for param_string in param_strings:
            print(param_string)
            param_path = os.path.join(selection_method_path, param_string)

            for test_pct in inst.get_list(sname, 'test_pcts'):
                test_pct_path = os.path.join(param_path, 'test_pct_' + str(test_pct))
                for train_pct in inst.get_list(sname, 'train_pcts'):
                    train_pct_path = os.path.join(test_pct_path, 'train_pct_' + str(train_pct))

                    #For each train pct, combine the RMSE's of each replica and print this to 
                    # a file along with test_pct and train_pct.
                    ref_energies = np.array([])
                    pred_energies = np.array([])

                    for test_num in range(int(inst.get(sname, 'test_num'))):
                        test_num_path = os.path.join(train_pct_path, 'test_num_' + str(test_num))
                        for train_num in range(int(inst.get(sname, 'train_num'))):
                            train_num_path = os.path.join(test_num_path, 'train_num_' + str(train_num))

                            test_results_path = os.path.join(train_num_path, 'test_results')
                            ref_energies_single_file, pred_energies_single_file = \
                                       get_ref_and_pred_energies_from_test_results(test_results_path)

                            ref_energies = np.append(ref_energies, ref_energies_single_file)
                            pred_energies = np.append(pred_energies, pred_energies_single_file)
                    print(test_pct, train_pct)
                    print(len(pred_energies))
                    print(ref_energies)
                    if len(pred_energies) != 0:
                        rmse = sqrt(mean_squared_error(ref_energies, pred_energies))
                        file_utils.write_row_to_csv('learning_curve_' + selection_method + '_' + param_string + '_test_pct_' + str(test_pct) + '.csv', [train_pct, rmse])
main()
