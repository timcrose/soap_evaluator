import os, glob, json, shutil

input_str = 'Target13_GA_jsons_no_dups_RDF_2124'
scpt_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(scpt_dir, input_str)
output_dir = os.path.join(scpt_dir, 'Target13_GA_jsons_no_dups')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

outfile_name = os.path.join(output_dir, input_str + '.xyz')

#Energy name in jsons
energy_name = 'energy'
#Structs with energies greater than energy_cutoff will not be included
# in the final xyz file.
energy_cutoff = 


#clean for new run
if os.path.isfile(outfile_name):
    os.remove(outfile_name)
    
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
GM_energy = get_GM_energy(data_files)
        
for data_file in data_files:
    data = read_json(data_file)
    en=data['properties'][energy_name]
    if (en - GM_energy < energy_cutoff) :
        energy_str = 'energy=' + str(en - GM_energy)
        N_atoms = str(len(data['geometry']))
        lattice_vecs_str = 'Lattice=' + '"' + str(data['properties']['lattice_vector_a'][0]) + ' ' + str(data['properties']['lattice_vector_a'][1]) + ' ' + str(data['properties']['lattice_vector_a'][2]) + ' ' + str(data['properties']['lattice_vector_b'][0]) + ' ' + str(data['properties']['lattice_vector_b'][1]) + ' ' + str(data['properties']['lattice_vector_b'][2]) + ' ' + str(data['properties']['lattice_vector_c'][0]) + ' ' + str(data['properties']['lattice_vector_c'][1]) + ' ' + str(data['properties']['lattice_vector_c'][2]) + '"'
        properties_str = 'Properties=species:S:1:pos:R:3'
        comment_line = lattice_vecs_str + ' ' + energy_str + ' ' + properties_str
        outfile_name = os.path.join(output_dir, input_str + '.xyz')
        with open(outfile_name, 'a') as f:
            f.write(N_atoms + '\n')
            f.write(comment_line + '\n')
            for xyzS in data['geometry']:
                f.write(str(xyzS[3]) + ' ' + str(xyzS[0]) + ' ' + str(xyzS[1]) + ' ' + str(xyzS[2]) + '\n')
