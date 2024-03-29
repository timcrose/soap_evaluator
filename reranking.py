import sys, glob, os
import numpy as np
from scipy import stats
import instruct
sys.path.append(os.path.join(os.environ["HOME"], "python_utils"))
import file_utils2 as file_utils

def sort_by_col(data, col):
    '''
    data: numpy array, shape (at least one column)
        array to sort

    col: int
        column index to sort

    return: the data sorted by the specified column

    purpose: Sort data by the specified column
    '''
    try:
        sorted_data = data[np.argsort(data[:,col])]
    except:
        return None
    return sorted_data

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

def get_ref_and_test_rankings(selection_method, param_string, test_energies, ref_energies, test_num_structs):
    #Sort by energy
    ref_energies = sort_by_col(ref_energies, col=1)
    test_energies = sort_by_col(test_energies, col=1)
    if ref_energies is None or test_energies is None:
        return None, None
    n_rows = ref_energies.shape[0]
    #print('n_rows', n_rows)
    #print('ref_energies', ref_energies)
    #print('test_energies', test_energies)

    save_path = 'ranking_list_raw_' + selection_method + '_' + param_string + '_test_num_structs_' + str(test_num_structs) + '.csv'
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
    try:
        line = lines[0].split()
    except:
        print('test_results_fname', test_results_fname, 'couldnt split first line')
        return None, None
    ref_energies = None
    pred_energies = None
    j = 0
    while line[0] != '0' and j < len(lines):
        if '#' in line[0]:
            j += 1
            line = lines[j].split()
        else:
            print('non-comment or result line in test_results_fname', test_results_fname, 'something is probably wrong.')
            return None, None
    if j == len(lines):
        print('EOF reached, something is probably wrong')
        return None, None
    i = 0
    while line[0] == str(i):
        ref_energies = add_element_to_array(ref_energies, [[i, float(line[1])]])
        pred_energies = add_element_to_array(pred_energies, [[i, float(line[2])]])
        i += 1
        j += 1
        line = lines[j].split()
    return ref_energies, pred_energies

def get_num_structs_in_xyz_file(path):
    return len(file_utils.grep('Lattice', path))

def main():
    inst_path = sys.argv[-1]
    inst = instruct.Instruct()
    inst.load_instruction_from_file(inst_path)

    owd = os.getcwd()
    sname = 'cross_val'
    for selection_method in inst.get_list(sname, 'selection_methods'):
        selection_method_path = os.path.join(owd, selection_method)
        param_strings = [os.path.basename(param_path) for param_path in file_utils.glob(os.path.join(selection_method_path, '*'))]
        for param_string in param_strings:
            param_path = os.path.join(selection_method_path, param_string)

            test_num_structs_paths = file_utils.glob(os.path.join(param_path, '*'))
            for test_num_structs_path in test_num_structs_paths:
                test_num_structs = int(os.path.basename(test_num_structs_path).split('_')[-1])
                train_num_structs_paths = file_utils.glob(os.path.join(test_num_structs_path, '*'))
                for train_num_structs_path in train_num_structs_paths:
                    train_num_structs = int(os.path.basename(train_num_structs_path).split('_')[-1]) 
                    #total_num_structs = get_num_structs_in_xyz_file(path=inst.get('cross_val', 'all_xyz_structs_fname'))
                    #For each train num_structs, average the R^2 of each replica and print this to 
                    # a file along with test_num_structs and train_num_structs.
                    r_sqrds = np.array([])

                    test_num_paths = file_utils.glob(os.path.join(train_num_structs_path, '*'))
                    for test_num_path in test_num_paths:
                        train_num_paths = file_utils.glob(os.path.join(test_num_path, '*'))
                        for train_num_path in train_num_paths:
                            test_results_path = os.path.join(train_num_path, 'test_results')
                            ref_energies, pred_energies = \
                                       get_ref_and_pred_energies_from_test_results(test_results_path)
                            if ref_energies is None or pred_energies is None:
                                continue
                            test_ranking, ref_ranking = get_ref_and_test_rankings(selection_method, param_string, pred_energies, ref_energies, test_num_structs)
                            if test_ranking is None or ref_ranking is None:
                                continue
                            r_sqrd = get_r_sqrd_from_rankings(test_ranking, ref_ranking)
                            r_sqrds = np.append(r_sqrds, r_sqrd)
                    if pred_energies is not None and len(pred_energies) != 0:
                        mean_r_sqrd = np.mean(r_sqrds)
                        file_utils.write_row_to_csv('average_kernel_reranking_' + selection_method + '_' + param_string + '_test_num_structs_' + str(test_num_structs) + '.csv', [train_num_structs, mean_r_sqrd])
main()

