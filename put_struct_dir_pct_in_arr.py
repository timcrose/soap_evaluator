from python_utils import list_utils, file_utils
from copy import deepcopy
import numpy as np


def put_struct_dir_pct_in_arr(test_struct_dir_pct_lst, train_struct_dir_pct_lst, struct_dir_lst, num_structs_to_use, data_files_matrix=None):
    '''
    test_struct_dir_pct_lst: list of float
        Each element represents the percentage (0-1) that the directory 
        of structures with corresponding index should approximately comprise in the
        first 20% of the resulting data files list used. sum(struct_dir_pct_lst)
        must be 1. If num_structs_to_use is too smalle, there is the possibility
        of 0 structures from a certain directory ending up in the first 20% even if
        its pct is > 0.

    train_struct_dir_pct_lst: list of float
        Each element represents the percentage (0-1) that the directory 
        of structures with corresponding index should approximately comprise in the
        last 80% of the resulting data files list used. sum(struct_dir_pct_lst)
        must be 1. If num_structs_to_use is too smalle, there is the possibility
        of 0 structures from a certain directory ending up in the last 80% even if
        its pct is > 0.

    struct_dir_lst: list of str
        List of directory paths where each directory contains structure files.
        Currently, it extracts just json structure files.

    num_structs_to_use: int
        Number of overall structures to include in the kernel.

    data_files_matrix: list of list of str, or None
        Each element of the outer list is a list of structure json files contained in a directory.
        The order of the outer list must correspond to the order of struct_dir_lst.
        If None: compute this matrix using the find function.

    Return: data_files, data_files_idx
    data_files: list of str
        Each element is a structure file path. The first 20% of structures are composed
        of approximately the percentages in struct_dir_pct_lst. These first 20% are suitable
        for selecting struct_dir_pct_lst structures when constructing a test sample of structures.
        The same can be said for the last 80% of structures for the training sample.
    data_files_idx: list of int
        Each element is the corresponding directory ID for each element of data_files. The first
        directory in struct_dir_lst has ID 0, the next dir in that list has ID 1, and so on.
        This will be useful for when sampling, we can know which elements of data files and of the
        kernel belong to which directory so we can select pct of them for training or testing.

    Purpose: If you want to select a certain percentage of structures for a given directory of structures
        to be included in the test universe (first 20% of the num_structs_to_use data files) and in the
        train universe (last 80%) then this function will work for that purpose unless the number of
        structures in a particular directory is insufficient or num_structs_to_use is too small
        (probably < 100 is too small).
    '''
    if data_files_matrix is None:
        data_files_matrix =  [file_utils.find(struct_dir, '*.json') for struct_dir in struct_dir_lst]

    num_data_files_in_each_dir = [len(data_files_matrix[i]) for i in range(len(data_files_matrix))]
    data_file_idx_lst = np.array(list_utils.flatten_list([[i] * len(data_files_matrix[i]) for i in range(len(data_files_matrix))]))

    data_files = np.array(list_utils.flatten_list(data_files_matrix))

    if not isinstance(num_structs_to_use, int):
        if num_structs_to_use == 'all':
            num_structs_to_use = sum(num_data_files_in_each_dir)
        else:
            raise ValueError('num_structs_to_use should be "all" or int but type(num_structs_to_use) is {} and has value {}'.format(type(num_structs_to_use),num_structs_to_use))
    ntest_univ = int(num_structs_to_use * 0.2)
    ntrain_univ = num_structs_to_use - ntest_univ

    if np.allclose(ntest_univ, 0, 1e-7, 1e-7):
        raise ValueError('ntest_univ = {} but should not be 0'.format(ntest_univ))
    if np.allclose(ntrain_univ, 0, 1e-7, 1e-7):
        raise ValueError('ntrain_univ = {} but should not be 0'.format(ntrain_univ))

    for i,pct in enumerate(test_struct_dir_pct_lst):
        assert(pct >= 0)
        if pct > 0 and int(pct * ntest_univ) == 0:
            raise ValueError('Provide higher ntest_univ and/or higher pct for struct dir {}. pct provided: {}'.format(struct_dir_lst[i], pct))
        elif int(pct * ntest_univ) > len(data_files_matrix[i]):
            raise ValueError('Provide lower pct than {} for struct dir {} or lower num_structs_to_use or put more structures in that dir because currently int(pct * ntest_univ) = {} is > len(data_files_matrix[i]) = {}'.format(pct, struct_dir_lst[i], int(pct * ntest_univ), len(data_files_matrix[i])))
    if not np.allclose(sum(test_struct_dir_pct_lst), 1, 1e-7,1e-7):
        raise ValueError('sum(test_struct_dir_pct_lst) != 1')


    for i,pct in enumerate(train_struct_dir_pct_lst):
        assert(pct >= 0)
        if pct > 0 and int(pct * ntrain_univ) == 0:
            raise ValueError('Provide higher ntrain_univ and/or higher pct for struct dir {}. pct provided: {}'.format(struct_dir_lst[i], pct))
        elif int(pct * ntrain_univ) > len(data_files_matrix[i]):
            raise ValueError('Provide lower pct than {} for struct dir {} or lower num_structs_to_use or put more structures in that dir because currently int(pct * ntrain_univ) = {} is > len(data_files_matrix[i]) = {}'.format(pct, struct_dir_lst[i], int(pct * ntrain_univ), len(data_files_matrix[i])))
    if not np.allclose(sum(train_struct_dir_pct_lst), 1, 1e-7, 1e-7):
        raise ValueError('sum(train_struct_dir_pct_lst) = {} but should be 1'.format(sum(train_struct_dir_pct_lst)))

    num_structs_assigned_lst = []
    for pct in test_struct_dir_pct_lst:
        num_structs_assigned_lst.append(int(pct * ntest_univ))
    num_leftover_structs = ntest_univ - sum(num_structs_assigned_lst)
    for i in range(len(struct_dir_lst)):
        if num_structs_assigned_lst[i] + num_leftover_structs <= len(data_files_matrix[i]):
            num_structs_assigned_lst[i] = num_structs_assigned_lst[i] + num_leftover_structs
            break

    for i,pct in enumerate(train_struct_dir_pct_lst):
        num_structs_assigned_lst[i] = num_structs_assigned_lst[i] + int(pct * ntrain_univ)
    num_leftover_structs = num_structs_to_use - sum(num_structs_assigned_lst)
    for i in range(len(struct_dir_lst)):
        if num_structs_assigned_lst[i] + num_leftover_structs <= len(data_files_matrix[i]):
            num_structs_assigned_lst[i] = num_structs_assigned_lst[i] + num_leftover_structs
            break

    if not np.allclose(sum(num_structs_assigned_lst),  num_structs_to_use, 1e-7, 1e-7):
        raise ValueError('sum(num_structs_assigned_lst) = {} should be equal to num_structs_to_use = {} but it is not.'.format(
                            sum(num_structs_assigned_lst), num_structs_to_use))
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
    for i in range(len(test_struct_dir_pct_lst)):
        # Get number of data files of this data dir that will end up in the test universe
        num_data_files_in_test_univ.append(int(test_struct_dir_pct_lst[i] * ntest_univ))

    num_leftover_test_structs = ntest_univ - sum(num_data_files_in_test_univ)
    while num_leftover_test_structs > 0:
        for i in range(len(num_data_files_in_test_univ)):
            if test_struct_dir_pct_lst[i] > 0:
                num_data_files_in_test_univ[i] = num_data_files_in_test_univ[i] + 1
                num_leftover_test_structs = ntest_univ - sum(num_data_files_in_test_univ)
                if np.allclose(num_leftover_test_structs, 0, 1e-7, 1e-7):
                    break
        num_leftover_test_structs = ntest_univ - sum(num_data_files_in_test_univ)

    for i in range(len(num_data_files_in_test_univ)):
        if num_data_files_in_test_univ[i] > list(data_files_idx_selected).count(i):
            raise ValueError('num_data_files_in_test_univ {} wants {} elements of dir {} but data_files_selected {} has {} elements of dir {}'.format(
                             num_data_files_in_test_univ, num_data_files_in_test_univ[i], i, list(data_files_idx_selected), list(data_files_idx_selected).count(i), i))

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
            if len(test_choice) + len(test_selection) > ntest_univ:
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

    if not np.allclose(len(test_selection), ntest_univ, 1e-7, 1e-7):
        raise ValueError('len(test_selection) = {} should be equal to ntest_univ = {}, but it is not.'.format(
                            len(test_selection), ntest_univ))

    if not np.allclose(len(test_selection) + len(train_selection), num_structs_to_use, 1e-7, 1e-7):
        raise ValueError('len(test_selection) + len(train_selection) = {} should be equal to num_structs_to_use = {}, but it is not.'.format(
                            len(test_selection) + len(train_selection), num_structs_to_use))
    if not np.allclose(len(test_selection_idx) + len(train_selection_idx), num_structs_to_use, 1e-7, 1e-7):
        raise ValueError('len(test_selection_idx) + len(train_selection_idx) = {} should be equal to num_structs_to_use = {}, but it is not.'.format(
                            len(test_selection_idx) + len(train_selection_idx), num_structs_to_use))

    return test_selection + train_selection, test_selection_idx + train_selection_idx
    

if __name__ == '__main__':
    struct_dir_lst = ['hi', 'bye', 'yo']
    data_files_matrix =  [['Qi', 'Wi', 'Ei', 'Ri', 'T', 'Y', 'U', 'I', 'O', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C'],
                          ['byea', 'byeb', 'byerb', 'byet', 'byeg', 'brg', 'bdg', 'brs', 'byes', 'bws', 'bjk', 'ber', 'bew', 'bq'],
                          ['you', 'yom', 'yum', 'yet', 'yot', 'yut', 'yog', 'yor', 'ywe']]
    num_structs_to_use = 26
    test_struct_dir_pct_lst = [0.5, 0.3, 0.2]
    train_struct_dir_pct_lst = [0.8, 0.2, 0]

    data_files, data_files_idx = put_struct_dir_pct_in_arr(test_struct_dir_pct_lst, train_struct_dir_pct_lst, struct_dir_lst, num_structs_to_use, data_files_matrix=data_files_matrix)
    print('data_files', data_files)
    print('data_files_idx', data_files_idx)