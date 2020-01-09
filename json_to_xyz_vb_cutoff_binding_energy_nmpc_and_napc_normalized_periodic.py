import os, glob, json, shutil, sys
import numpy as np
from python_utils import file_utils
from copy import deepcopy


def read_json(file_name):
    with open (file_name) as in_file:
        data = json.load(in_file)
        return data


def get_GM_energy(all_data, energy_name):
    GM_energy = 0
    for data in all_data:
        en= float(data['properties'][energy_name])
        if en < GM_energy:
            GM_energy = en
    return GM_energy


def binding_energy(nmpc, total_energy, single_molecule_energy):
    return total_energy - (nmpc * single_molecule_energy)


def normalized_BE_by_napc(napc, nmpc, total_energy, single_molecule_energy, BE=None):
    if BE is None:
        BE = binding_energy(nmpc, total_energy, single_molecule_energy)
    return BE / napc


def make_supercells(data_files, radius=15.0, num_atoms=None):
    '''
    data_files: list of str
        list of paths to json files that contain structures
    radius: float or None
        distance in Ang from the origin along the a,b,c directions above which to not create another
        cell in that direction (though a partial cell may extend beyond radius i.e. round up).
        Cells will then be filled in to form the overall supercell. If radius is None, then num_atoms 
        will be used alone to determine the number of cells in the supercell instead.
    num_atoms: int, str, or None
        If int, then will create supercells in order to have the specified number of atoms in the supercell or
        fail if cannot. If None, then radius alone will be used to determine the
        number of cells in the supercell instead.

    Purpose: Convert geometries in data_files to ones with supercells to try 
        to account for periodicity. The supercells are created to enable the same
        number of atoms in the supercell of all structures.

    Notes: num_atoms and radius cannot both be None or both be numbers.
    
    Return:
    struct_data: list of data dicts for structures now with supercell geometry properties.
    '''
    if type(num_atoms) != float and type(num_atoms) != int and num_atoms is not None:
        raise TypeError('num_atoms must be float, int, or None')
    if type(radius) != float and type(radius) != int and radius is not None:
        raise TypeError('radius must be float, int, or None')
    if num_atoms is None and radius is None:
        raise Exception('If num_atoms is None, then radius must be a number.')
    if (type(num_atoms) is float or type(num_atoms) is int) and (type(radius) is float or type(radius) is int):
        raise Exception('num_atoms and radius cannot both be numbers because it is unclear which to choose.')
    all_data = []
    for data_file in data_files:
        data = read_json(data_file)
        geometry = np.array([data['geometry'][i][:3] for i in range(len(data['geometry']))])
        original_geo = deepcopy(geometry)
        original_species = [data['geometry'][i][3] for i in range(len(data['geometry']))]
        napc = len(original_geo)
        lattice_vector_a = np.array(data['properties']['lattice_vector_a'])
        lattice_vector_b = np.array(data['properties']['lattice_vector_b'])
        lattice_vector_c = np.array(data['properties']['lattice_vector_c'])
        if radius is None:
            if num_atoms % napc != 0:
                raise Exception('Cannot create supercell with ', num_atoms, ' atoms from unit cell with ', napc, 'atoms')
            if num_atoms == napc:
                continue
            num_cells = int(num_atoms / napc)
            if num_cells**(1.0/3.0) % 2 != 0:
                raise Exception('Must have equal number of cells in the a, b, and c directions forming a supercube (except not necessarily 90 degrees)',
                                'num_cells**(1.0/3.0) % 2 = ', num_cells**(1.0/3.0) % 2)
            num_cells_in_a_direction = int(num_cells**(1.0/3.0))
            num_cells_in_b_direction = int(num_cells**(1.0/3.0))
            num_cells_in_c_direction = int(num_cells**(1.0/3.0))
        else:
            radius = float(radius)
            num_cells_in_a_direction = int(np.ceil(radius / np.linalg.norm(lattice_vector_a)))
            num_cells_in_b_direction = int(np.ceil(radius / np.linalg.norm(lattice_vector_b)))
            num_cells_in_c_direction = int(np.ceil(radius / np.linalg.norm(lattice_vector_c)))
            num_cells = num_cells_in_a_direction * num_cells_in_b_direction * num_cells_in_c_direction

        print('num_cells', num_cells)
        #print('num_cells_in_a_direction', num_cells_in_a_direction,
        #        'num_cells_in_b_direction', num_cells_in_b_direction,
        #        'num_cells_in_c_direction', num_cells_in_c_direction)
        
        for i in range(num_cells_in_a_direction):
            for j in range(num_cells_in_b_direction):
                for k in range(num_cells_in_c_direction):
                    if i == j == k == 0:
                        continue
                    geometry = np.vstack((geometry, original_geo + lattice_vector_a * i + lattice_vector_b * j + lattice_vector_c * k))
        # get repeated species
        species = original_species * num_cells
        # put the supercell into the dictionary
        data['geometry'] = [list(row) + [species[j]] for j, row in enumerate(geometry)]
        all_data.append(data)

    return all_data


def main():
    #Structs with energies greater than energy_cutoff will not be included
    # in the final xyz file.
    energy_cutoff = 'NaN'
    #num_structs = 1045
    nmpc = 2.0
    #napm = 11.0 # target2 
    #napm = 12.0 # target13
    napm = 13.0 # target1, FUQJIK
    napc = nmpc * napm
    #single molecule energy is Total energy converged relaxed SCF in this case.
    #single_molecule_energy = -19594.403745344 # target2
    #single_molecule_energy = -164356.962622325 # target13
    single_molecule_energy = -8357.500268932 # target1
    #single_molecule_energy = -151601.668571928 # FUQJIK
    #input_str is path to folder with jsons with energies
    #input_str = '/home/trose/SPE_evaluations/target2/2mpc/output'
    #input_str = '/home/trose/genarris_run_calcs/target2/soap/target2_4mpc_initial_geos'
    #input_str = '/home/trose/SPE_evaluations/target13/2mpc/gen_more/target13_2mpc_jsons_more'
    #input_str = '/home/trose/SPE_evaluations/target2/2mpc/gen_more/target2_2mpc_jsons_more'
    #input_str = '/home/trose/SPE_evaluations/target2/4mpc/target2_4mpc_jsons_more'
    #input_str = '/home/trose/SPE_evaluations/target13/4mpc/target13_4mpc_raw_pool_jsons_no_dups_SPE_energies'
    #input_str = '/home/trose/genarris_run_calcs/target13/4mpc/target13_4mpc_raw_pool_jsons'
    #input_str = '/home/trose/test_glosim/target1/get_sii/test_structs'
    input_str = '/home/trose/genarris_run_calcs/target1/2mpc/beta_genarris/target1_2mpc_with_energies'
    #input_str = '/home/trose/genarris_run_calcs/target2/2mpc/target2_2mpc_with_energies'
    scpt_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(scpt_dir, input_str)
    try:
        output_dir = os.path.join(scpt_dir, str(int(energy_cutoff)) + 'eV_cutoff/normalized_BE')
    except:
        output_dir = os.path.join(scpt_dir, str(energy_cutoff))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outfile_name = os.path.join(output_dir, 'target1_' + str(int(nmpc)) + 'mpc_SCF_energies_napc_normalized_BE_unrelaxed_periodic.xyz')

    #Energy name in jsons
    energy_name = 'energy'

    #clean for new run
    file_utils.rm(outfile_name)
        
    data_files = glob.glob(input_dir + '/*')

    print('making supercells')
    all_data = make_supercells(data_files) #, radius=None, num_atoms=napc * 2**3)
    print('got all_data')

    #Out of all the jsons, this is the lowest energy
    if type(energy_cutoff) is not str:
        energy_cutoff = 1e9
    GM_energy = get_GM_energy(all_data, energy_name)

    for data in all_data:
        if 'properties' in data and energy_name in data['properties']:
            en=data['properties'][energy_name]
        else:
            en = 'none'
        if en == 'none':
            energy_str = 'energy=' + en
        elif (en - GM_energy < energy_cutoff):
            energy_str = 'energy=' + str(normalized_BE_by_napc(napc, nmpc, en, single_molecule_energy))
        if en == 'none' or (en - GM_energy < energy_cutoff):
            N_atoms = str(len(data['geometry']))
            lattice_vecs_str = 'Lattice=' + '"' + str(data['properties']['lattice_vector_a'][0]) + ' ' + str(data['properties']['lattice_vector_a'][1]) + ' ' + str(data['properties']['lattice_vector_a'][2]) + ' ' + str(data['properties']['lattice_vector_b'][0]) + ' ' + str(data['properties']['lattice_vector_b'][1]) + ' ' + str(data['properties']['lattice_vector_b'][2]) + ' ' + str(data['properties']['lattice_vector_c'][0]) + ' ' + str(data['properties']['lattice_vector_c'][1]) + ' ' + str(data['properties']['lattice_vector_c'][2]) + '"'
            properties_str = 'Properties=species:S:1:pos:R:3'
            struct_id = data['struct_id']
            struct_id_str = 'struct_id=' + struct_id
            comment_line = lattice_vecs_str + ' ' + energy_str + ' ' + struct_id_str + ' ' + properties_str
            outfile_name = os.path.join(output_dir, outfile_name)
            
            with open(outfile_name, 'a') as f:
                f.write(N_atoms + '\n')
                f.write(comment_line + '\n')
                for xyzS in data['geometry']:
                    f.write(str(xyzS[3]) + ' ' + str(xyzS[0]) + ' ' + str(xyzS[1]) + ' ' + str(xyzS[2]) + '\n')
main()