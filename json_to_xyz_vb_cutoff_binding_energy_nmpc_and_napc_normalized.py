import os, glob, json, shutil, sys
from python_utils import file_utils

#Structs with energies greater than energy_cutoff will not be included
# in the final xyz file.
energy_cutoff = 'NaN'
#num_structs = 1045
nmpc = 2.0
napm = 11.0 # target2 
#napm = 12.0 # target13
#napm = 13.0 # target1, FUQJIK
napc = nmpc * napm
#single molecule energy is Total energy converged relaxed SCF in this case.
single_molecule_energy = -19594.403745344 # target2
#single_molecule_energy = -164356.962622325 # target13
#single_molecule_energy = -8357.500268932 # target1
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
#input_str = '/home/trose/genarris_run_calcs/target1/2mpc/beta_genarris/target1_2mpc_with_energies'
input_str = '/home/trose/genarris_run_calcs/target2/2mpc/target2_2mpc_with_energies'
scpt_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(scpt_dir, input_str)
try:
    output_dir = os.path.join(scpt_dir, str(int(energy_cutoff)) + 'eV_cutoff/normalized_BE')
except:
    output_dir = os.path.join(scpt_dir, str(energy_cutoff))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

outfile_name = os.path.join(output_dir, 'target2_' + str(int(nmpc)) + 'mpc_SCF_energies_napc_normalized_BE_unrelaxed.xyz')

#Energy name in jsons
energy_name = 'energy'

#clean for new run
file_utils.rm(outfile_name)
    
data_files = glob.glob(input_dir + '/*')

def get_GM_energy(data_files):
    GM_energy = 0
    for data_file in data_files:
        data = read_json(data_file)
        en= float(data['properties'][energy_name])
        if en < GM_energy:
            GM_energy = en

    return GM_energy

def read_json(file_name):
    with open (file_name) as in_file:
        data = json.load(in_file)
        return data

#Out of all the jsons, this is the lowest energy
if type(energy_cutoff) is not str:
    energy_cutoff = 1e9
GM_energy = get_GM_energy(data_files)

def binding_energy(nmpc, total_energy, single_molecule_energy):
    return total_energy - (nmpc * single_molecule_energy)

def normalized_BE_by_napc(napc, nmpc, total_energy, single_molecule_energy, BE=None):
    if BE is None:
        BE = binding_energy(nmpc, total_energy, single_molecule_energy)
    return BE / napc

for data_file in data_files:
    data = read_json(data_file)
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
