import os, sys, instruct
import numpy as np
sys.path.append(os.environ['HOME'])
from python_utils import file_utils

def get_struct_line_idx_in_xyz(xyz_fpath):
    _, line_nums = file_utils.grep('Lattice', xyz_fpath, return_line_nums=True, return_fpaths=False)
    return np.array(line_nums) - 1


def separate_xyz_into_segments(xyz_fpath, num_segments, output_dir):
    '''
    xyz_fpath: str
        path to .xyz file that has many structures concatenated
    num_segments: int
        Number of xyzs to write where each one will be that fraction of the
        total number of structures in xyz_fpath.
        e.g. if num_segments is 10 and the number of structures in xyz_fpath
        is 10000, then 10 xyz files will be written and each will have 1000
        structures in it.
    output_dir: str
        Path of directory to put the segmented xyz files in.

    Return: None

    Purpose: Calculation of a kernel with many structures in the xyz file
        can take way too long. So, break up the large xyz file into many
        (num_segments) smaller xyz files so a different process can 
        calculate a kernel for each of the smaller xyz files. Then, the
        smaller kernels will be combined into a larger kernel if needed. 
        This means there will be redundant calculations, and the higher
        num_segments is, the greater number of redundant calculations
        there will be.
        The first int(num_structs_in_total_xyz / num_segments) structs will
        be put in the first xyz file indexed by a '0', the next
        int(num_structs_in_total_xyz / num_segments) structs will be put 
        in the second xyz file indexed by a '1', and so on. The last
        xyz file may have less than 
        int(num_structs_in_total_xyz / num_segments) structs in it if
        num_structs_in_total_xyz / num_segments does not have a 
        remainder of 0.
    '''
    dataset_lines = file_utils.get_lines_of_file(xyz_fpath)
    struct_line_idxs = get_struct_line_idx_in_xyz(xyz_fpath)
    num_structs_in_total_xyz = float(len(struct_line_idxs))
    num_structs_per_segment = int(num_structs_in_total_xyz / float(num_segments))
    xyz_fname = file_utils.fname_from_fpath(xyz_fpath)
    file_utils.mkdir_if_DNE(output_dir)
    for i in range(int(num_segments)):
        output_path = os.path.join(output_dir, xyz_fname + '_' + str(i) + '.xyz')
        print('output_path',output_path)
        if (i + 1) * num_structs_per_segment >= len(struct_line_idxs):
            lines_in_this_segment = dataset_lines[struct_line_idxs[i * num_structs_per_segment] : ]
        else:
            lines_in_this_segment = dataset_lines[struct_line_idxs[i * num_structs_per_segment] : struct_line_idxs[(i + 1) * num_structs_per_segment]]
        file_utils.write_lines_to_file(output_path, lines_in_this_segment, mode='w')

    
def main():
    inst_path = sys.argv[-1]
    inst = instruct.Instruct()
    inst.load_instruction_from_file(inst_path)

    sname = 'separate_xyz_into_segments'
    num_segments = inst.get_eval(sname, 'num_segments')
    xyz_fpath = inst.get(sname, 'xyz_fpath')
    output_dir = inst.get(sname, 'output_dir')
    print('output_dir', output_dir)
    separate_xyz_into_segments(xyz_fpath, num_segments, output_dir)

if __name__ == '__main__':
    main()