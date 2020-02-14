from python_utils import list_utils, file_utils
from copy import deepcopy
import numpy as np


def put_struct_dir_pct_in_arr(struct_dir_pct_lst, struct_dir_lst, num_structs_to_use):
    '''
    struct_dir_pct_lst: list of float
        Each element represents the percentage (0-1) that the directory 
        of structures with corresponding index should comprise in the
        resulting data files used. This percentage will also try to
        be populated in the test and train structures sets, but
        this is not guaranteed if the number of structures is too
        small to fill the request or the num_structs_to_use is
        too small. Note that the number of structures selected for
        data files and test and train partitions will also only
        approximately be the requested percentage if num_structs_to_use
        * 0.2 (ntest_univ) * pct is a counting number. sum(struct_dir_pct_lst)
        must be 1.

    struct_dir_lst: list of str
        List of directory paths where each directory contains structure files.
        Currently, it extracts just json structure files.

    num_structs_to_use: int
        Number of overall structures to include in the kernel.

    Return: data_files
    data_files: list of str
        Each element is a structure file path. The first 20% of structures are composed
        of approximately the percentages in struct_dir_pct_lst. These first 20% are suitable
        for selecting struct_dir_pct_lst structures when constructing a test sample of structures.
        The same can be said for the last 80% of structures for the training sample.

    Purpose: If you want to select a certain percentage of structures for a given directory of structures
        to be included in the test universe (first 20% of the num_structs_to_use data files) and in the
        train universe (last 80%) then this function will work for that purpose unless the number of
        structures in a particular directory is insufficient or num_structs_to_use is too small
        (probably < 100 is too small).
    '''

    data_files_matrix =  [file_utils.find(struct_dir, '*.json') for struct_dir in struct_dir_lst]

    num_data_files_in_each_dir = [len(data_files_matrix[i]) for i in range(len(data_files_matrix))]
    data_file_idx_lst = np.array(list_utils.flatten_list([[i] * len(data_files_matrix[i]) for i in range(len(data_files_matrix))]))

    data_files = np.array(list_utils.flatten_list(data_files_matrix))

    if sum(struct_dir_pct_lst) != 1:
        raise ValueError('sum(struct_dir_pct_lst) != 1')
    
    ntest_univ = int(num_structs_to_use * 0.2)
    ntrain_univ = num_structs_to_use - ntest_univ

    assert(ntest_univ != 0)

    num_structs_assigned_lst = []
    for pct in struct_dir_pct_lst:
        num_structs_assigned_lst.append(int(pct * num_structs_to_use))
    num_leftover_structs = num_structs_to_use - sum(num_structs_assigned_lst)
    for i in range(len(struct_dir_lst)):
        if num_structs_assigned_lst[i] + num_leftover_structs <= len(data_files_matrix[i]):
            num_structs_assigned_lst[i] = num_structs_assigned_lst[i] + num_leftover_structs
            break
    assert(sum(num_structs_assigned_lst) <= num_structs_to_use)
    data_files_selected = []
    data_files_idx_selected = []
    for i in range(len(struct_dir_lst)):
        choices = np.random.choice(np.arange(sum(num_data_files_in_each_dir[:i]), sum(num_data_files_in_each_dir[:i + 1])), num_structs_assigned_lst[i], replace=False)
        if len(choices) > 0:

            data_files_selected += list(data_files[choices])
            data_files_idx_selected += list(data_file_idx_lst[choices])
    data_dir_idx_set = set(data_files_idx_selected)
    data_files_selected = np.array(data_files_selected)
    data_files_idx_selected = np.array(data_files_idx_selected)

    num_data_files_in_test_univ = []
    for i in range(len(struct_dir_pct_lst)):
        # Get number of data files of this data dir that will end up in the test universe
        num_data_files_in_test_univ.append(int(struct_dir_pct_lst[i] * ntest_univ))

    num_leftover_test_structs = ntest_univ - sum(num_data_files_in_test_univ)
    while num_leftover_test_structs > 0:
        for i in range(len(num_data_files_in_test_univ)):
            if struct_dir_pct_lst[i] > 0:
                num_data_files_in_test_univ[i] = num_data_files_in_test_univ[i] + 1
        num_leftover_test_structs = ntest_univ - sum(num_data_files_in_test_univ)

    for i in range(len(num_data_files_in_test_univ)):
        assert(len(np.where(num_data_files_in_test_univ == i)[0]) >= len(np.where(data_files_selected == i)[0]))

    # Choose num_data_files_in_test_univ[i] random structures out of the ones selected overall to be
    # in the test universe. The rest will go in the train universe.
    test_selection = []
    test_selection_idx = []
    train_selection = []
    train_selection_idx = []
    for i in range(len(num_data_files_in_test_univ)):
        where_dir_i_structs = np.arange(len(data_files_idx_selected))[np.where(data_files_idx_selected == i)]
        test_choice = np.random.choice(where_dir_i_structs, num_data_files_in_test_univ[i], replace=False)
        if len(test_choice) > 0 and len(test_selection) < ntest_univ:
            if len(test_choice) + len(test_selection) <= ntest_univ:
                test_choice = test_choice[:ntest_univ - len(test_selection)]
            test_selection += list(data_files_selected[test_choice])
            test_selection_idx += list(data_files_idx_selected[test_choice])
        elif len(test_selection) >= ntest_univ:
            test_choice = []
        train_choice_set = set(where_dir_i_structs) - set(test_choice)
        train_choice = np.array(list(deepcopy(train_choice_set)))
        if len(train_choice) > 0:
            train_selection += list(data_files_selected[train_choice])
            train_selection_idx += list(data_files_idx_selected[train_choice])

    assert(len(test_selection) == ntest_univ)
    assert(len(test_selection) + len(train_selection) == num_structs_to_use)
    return test_selection + train_selection


if __name__ == '__main__':
    struct_dir_pct_lst = [0, 0.8, 0.2]
    struct_dir_lst = ['hi', 'bye', 'yo']
    num_structs_to_use = 10
    data_files_matrix =  [['Qi', 'Wi', 'Ei', 'Ri','Fi'],
                        ['byea', 'byeb', 'byerb', 'byet', 'byeg','borg', 'blerg', 'burg', 'bort'],
                        ['you', 'yom','yrr', 'yor','yot','yet']]
    put_struct_dir_pct_in_arr(struct_dir_pct_lst, struct_dir_lst, num_structs_to_use)
