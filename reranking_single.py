import sys, glob, os
import numpy as np
from scipy import stats
import instruct
sys.path.append('/home/trose/python_utils')
import file_utils

def sort_by_col(data, col):
    '''
    data: numpy array, shape (at least one column)
        array to sort

    col: int
        column index to sort

    return: the data sorted by the specified column

    purpose: Sort data by the specified column
    '''
    return data[np.argsort(data[:,col])]

def add_element_to_array(data, el, axis=0):
    '''
    data: numpy array, any shape

    el: array-like
        element to add to array

    axis: int
        axis index to concatenate the array to

    return: array with element added

    purpose: Instantiate array if None or concatenate
        el to array if not None
    '''

    if data is None:
        data = np.array(el)
    else:
        data = np.concatenate([data, np.array(el)], axis=axis)

    return data

def get_ref_and_test_rankings(test_energies, ref_energies, test_num_structs):
    #Sort by energy
    ref_energies = sort_by_col(ref_energies, col=1)
    test_energies = sort_by_col(test_energies, col=1)

    n_rows = ref_energies.shape[0]
    #print('n_rows', n_rows)
    #print('ref_energies', ref_energies)
    #print('test_energies', test_energies)

    save_path = 'ranking_list_raw_test_num_structs_' + str(test_num_structs) + '.csv'
    ranking_list = [[np.where(ref_energies == r)[0][0], np.where(test_energies == r)[0][0]] for r in range(n_rows)]
    #print(ranking_list)
    file_utils.write_rows_to_csv(save_path, ranking_list, delimiter=' ')

    ranking_array = np.array(ranking_list)

    ref_ranking = ranking_array[:,:1].flatten()

    test_ranking = ranking_array[:,1:2].flatten()

    return test_ranking, ref_ranking

def get_r_sqrd_from_rankings(test_ranking, ref_ranking):
    slope, intercept, r_value, p_value, std_err = stats.linregress(test_ranking, ref_ranking)
    r_sqrd = r_value ** 2
    return r_sqrd

def get_ref_and_pred_energies_from_test_results(test_results_fname):
    if not os.path.isfile(test_results_fname):
        return np.array([]), np.array([])
    with open(test_results_fname) as f:
        lines = f.readlines()
    line = lines[0].split()
    ref_energies = None
    pred_energies = None
    i = 0
    while line[0] == str(i):
        ref_energies = add_element_to_array(ref_energies, [[i, float(line[1])]])
        pred_energies = add_element_to_array(pred_energies, [[i, float(line[2])]])
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

                #For each train num_structs, average the R^2 of each replica and print this to 
                # a file along with test_num_structs and train_num_structs.
                r_sqrds = np.array([])

                for test_num in range(int(inst.get(sname, 'test_num'))):
                    test_num_path = os.path.join(train_num_structs_path, 'test_num_' + str(test_num))
                    for train_num in range(int(inst.get(sname, 'train_num'))):
                        train_num_path = os.path.join(test_num_path, 'train_num_' + str(train_num))

                        test_results_path = os.path.join(train_num_path, 'test_results')
                        ref_energies, pred_energies = \
                                   get_ref_and_pred_energies_from_test_results(test_results_path)

                        test_ranking, ref_ranking = get_ref_and_test_rankings(pred_energies, ref_energies, test_num_structs)
                        r_sqrd = get_r_sqrd_from_rankings(test_ranking, ref_ranking)
                        r_sqrds = np.append(r_sqrds, r_sqrd)
                if len(pred_energies) != 0:
                    mean_r_sqrd = np.mean(r_sqrds)
                    file_utils.write_row_to_csv('average_kernel_reranking_' param_string + '_test_num_structs_' + str(test_num_structs) + '.csv', [train_num_structs, mean_r_sqrd])
main()

