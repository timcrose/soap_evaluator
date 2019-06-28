import sys, os
sys.path.append(os.path.join(os.environ['HOME'], 'python_utils'))
import file_utils

def create_associated_energy_dat_file(all_xyzs_dir, all_xmpc_struct_ids, all_kernels_dir, all_kernel_fpath):
    search_str = 'energy='
    all_xyz_fpaths = file_utils.glob(os.path.join(all_xyzs_dir, '*.xyz'))
    all_comment_lines = file_utils.grep(search_str, all_xyz_fpaths)
    all_energy_list = []
    for struct_id in all_xmpc_struct_ids:
        for comment_line in all_comment_lines:
            if struct_id in comment_line:
                comment_line = comment_line[comment_line.find(search_str) + len(search_str) :]
                energy = comment_line.split()[0]
                all_energy_list.append(energy)
                break
    if len(all_energy_list) != len(all_xmpc_struct_ids):
        raise Exception(len(all_energy_list), '= len(all_energy_list) != len(all_xmpc_struct_ids) =', len(all_xmpc_struct_ids))
    all_energy_lines = [en + '\n' for en in all_energy_list]
    energy_fpath = os.path.join(all_kernels_dir, 'en_all_for_' + file_utils.fname_from_fpath(all_kernel_fpath) + '.dat')
    file_utils.write_lines_to_file(energy_fpath, all_energy_lines, mode='w')

def create_xmpc_overall_kernel(x, all_kernels_dir, all_xyzs_dir, kernel_prefix):
    all_kernels = file_utils.glob(os.path.join(all_kernels_dir, '*.k'))
    all_xmpc_kernels = []
    all_xmpc_struct_ids = []
    for kernel in all_kernels:
        kernel = os.path.basename(kernel)
        struct_ids = kernel[:kernel.find('-')]
        #print('struct_ids', struct_ids)
        if len(struct_ids.split('._.')) != 2:
            continue
        struct_id_0, struct_id_1 = struct_ids.split('._.')
        #print('struct_id_0, struct_id_1', struct_id_0, struct_id_1)
        if str(x) + 'mpc' in struct_id_0 and str(x) + 'mpc' in struct_id_1:
            all_xmpc_kernels.append(kernel)
            if struct_id_0 not in all_xmpc_struct_ids:
                all_xmpc_struct_ids.append(struct_id_0)
            if struct_id_1 not in all_xmpc_struct_ids:
                all_xmpc_struct_ids.append(struct_id_1)
    overall_kernel_matrix = [[None for i in range(len(all_xmpc_struct_ids))] for j in range(len(all_xmpc_struct_ids))]
    #print('all_xmpc_struct_ids', all_xmpc_struct_ids)     
    i = 0
    while i < len(all_xmpc_struct_ids) - 1:
        for j, struct_id in enumerate(all_xmpc_struct_ids[i + 1 :]):
            pairwise_kernel_fpath = file_utils.glob(os.path.join(all_kernels_dir, all_xmpc_struct_ids[i] + '._.' + struct_id + '*.k'))
            if pairwise_kernel_fpath == []:
                pairwise_kernel_fpath = file_utils.glob(os.path.join(all_kernels_dir, struct_id + '._.' + all_xmpc_struct_ids[i] + '*.k'))
                if pairwise_kernel_fpath == []:
                    raise Exception('pairwise kernel fpath:', os.path.join(all_kernels_dir, all_xmpc_struct_ids[i] + '._.' + struct_id + '*.k'), 'not found')
            pairwise_kernel_fpath = pairwise_kernel_fpath[0]
            lines = file_utils.get_lines_of_file(pairwise_kernel_fpath)
            comment_line = lines[0]
            matrix_row_0 = list(map(float, lines[1].split()))
            matrix_row_1 = list(map(float, lines[2].split()))
            overall_kernel_matrix[i][i] = matrix_row_0[0]
            overall_kernel_matrix[i + 1 + j][i + 1 + j] = matrix_row_1[1]
            overall_kernel_matrix[i][i + 1 + j] = matrix_row_0[1]
            overall_kernel_matrix[i + 1 + j][i] = matrix_row_0[1]
        i += 1
    for row in overall_kernel_matrix:
        if None in row:
            raise Exception('None was found in overall_kernel_matrix. Need to calculate some more pairwise kernels.', overall_kernel_matrix)
    overall_kernel_matrix_writable_format = [' '.join(list(map(str, row))) + '\n' for row in overall_kernel_matrix]
    #print('overall_kernel_matrix_writable_format', overall_kernel_matrix_writable_format)
    #print('comment_line', comment_line)
    
    kernel_file_content = [comment_line if i == 0 else overall_kernel_matrix_writable_format[i - 1] for i in range(len(overall_kernel_matrix) + 1)]
    #print('kernel_file_content', kernel_file_content)
    pairwise_kernel_fname = file_utils.fname_from_fpath(pairwise_kernel_fpath, include_ext=True)
    all_kernel_fpath = os.path.join(all_kernels_dir, kernel_prefix + pairwise_kernel_fname[pairwise_kernel_fname.find('-'):])
    file_utils.write_lines_to_file(all_kernel_fpath, kernel_file_content, mode='w')
    # Write corresponding struct_id list to a file
    all_xmpc_struct_ids_lines = [struct_id + '\n' for struct_id in all_xmpc_struct_ids]
    all_xmpc_struct_ids_fpath = os.path.join(all_kernels_dir, 'struct_ids_for_' + file_utils.fname_from_fpath(all_kernel_fpath) + '.log')
    file_utils.write_lines_to_file(all_xmpc_struct_ids_fpath, all_xmpc_struct_ids_lines, mode='w')
    create_associated_energy_dat_file(all_xyzs_dir, all_xmpc_struct_ids, all_kernels_dir, all_kernel_fpath)

def main():
    all_kernels_dir = '/home/trose/glosim_20190211/run_calcs/soap_envs/target13/4mpc/tiny_test/generate_soap_envs/all_kernels'
    all_xyzs_dir = '/home/trose/glosim_20190211/run_calcs/soap_envs/target13/4mpc/tiny_test/generate_soap_envs/all_xyzs'
    create_xmpc_overall_kernel(4, all_kernels_dir, all_xyzs_dir, 'blargh_4mpc')

main()