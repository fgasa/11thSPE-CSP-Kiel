'''

__author__ = 'fgasa'

'''
import os
import argparse
import numpy as np
import pyvista as pv

CASE_CONFIG = {
    'spe11a': {
        'nI': 280 # Element number in I direction
    },
    'spe11b': {
        'nI': 842
    },
    'spe11c': {
        'nI': 170
    }
}

def get_path_with_extension(folder_path, extension):
    for file in os.listdir(folder_path):
        if file.endswith('.' + extension):
            file_path = os.path.join(folder_path, file)
            return file_path

    return None # Return None if no file with the specified extension is found

def read_mesh_file(filepath, ni):
    '''Read a mesh file and calculate cell data.'''
    mesh = pv.read(filepath)

    # Get the cell data
    cell_data = mesh.cell_data

    # Print the dimensions and shape of the mesh
    print(f' Msg: Number of points: {mesh.n_points}')
    print(f' Msg: Number of cells: {mesh.n_cells}')
    print(f' Msg: Mesh length: {mesh.length}')
    xyz_coordinates = mesh.points

    x_nodes = xyz_coordinates[:, 0]
    y_nodes = xyz_coordinates[:, 1]
    z_nodes = xyz_coordinates[:, 2]

    # Get the coordinates of the cell centers
    cell_centers = mesh.cell_centers().points
    x_elements = cell_centers[:, 0]
    y_elements = cell_centers[:, 1]
    z_elements = cell_centers[:, 2]

    tops = y_elements
    distance_x = x_elements

    # Store the new array as cell data within mesh
    mesh.cell_data['TOPS'] = tops.astype(np.int32)
    mesh.cell_data['DISTANCE_X'] = distance_x.astype(np.int32)
    mesh.cell_data['DISTANCE_Y'] = z_elements.astype(np.int32)

    print(cell_data)
    return mesh

def generate_region_index(mesh, ni, filename, report_dir, keyname):

    reversed_array = mesh['gmsh:physical'][::-1] #mesh['attribute'][::-1]

    # Flip/mirror every ni items
    flipped_array = np.concatenate(
    [reversed_array[i:i + ni][::-1] for i in range(0, len(reversed_array), ni)])

    # Reshape the array into 2 dimensions with 10 columns
    two_dimensional_array = flipped_array.reshape(-1, 20)
    output_file = os.path.join(report_dir, filename + '_' + keyname + '.INC')

    np.savetxt(output_file, two_dimensional_array, delimiter=' ', fmt='%d', header=keyname, footer='/',
                   comments='')
    print(f' Msg: Output file name: {output_file}')

def generate_petro_param(mesh, case_name, ni, filename, report_dir, keyname):
    values_to_replace = [1, 2, 3, 4, 5, 6, 7]
    reversed_array = mesh['gmsh:physical'][::-1]
    if case_name == 'spe11a':
        if keyname == 'PORO':
            replacement_values = [0.4400, 0.4300, 0.4400, 0.4500, 0.4300, 0.4600, 0.0000]
        elif keyname == 'PERMX':
            replacement_values = [40529.9999, 506624.9850, 1013249.9700, 2026499.9400, 4052999.8800, 10132499.7000, 0.0000]
        elif keyname == 'THCONR':
            replacement_values = [0, 0, 0, 0, 0, 0, 0]
    elif case_name == 'spe11b' or case_name == 'spe11c':

        if keyname == 'PORO':
            replacement_values = [0.1000, 0.2000, 0.2000, 0.2000, 0.2500, 0.3500, 1e-6]
        elif keyname == 'PERMX':
            replacement_values = [0.1013, 101.3250, 202.6500, 506.6250, 1013.2500, 2026.4999, 1e-6]
        elif keyname == 'THCONR':
            replacement_values = [164.1600, 108.0000, 108.0000, 108.0000, 79.4880, 22.4600, 172.8000]

    modified_array = np.select([reversed_array == value for value in values_to_replace], replacement_values,
                               default=reversed_array)

    flipped_array = np.concatenate(
        [modified_array[i:i + ni][::-1] for i in range(0, len(modified_array), ni)])

    multdim_array = flipped_array.reshape(-1, 20)

    output_file = os.path.join(report_dir, filename + '_' + keyname + '.INC')

    print(f' Msg: Output file name: {output_file}')

    np.savetxt(output_file, multdim_array, delimiter=' ', header=keyname, fmt='%0.6f', footer='/',
               comments='')

def main():
    parser = argparse.ArgumentParser(
        description='This script generates parameters for OPM Flow models based on structural mesh (.vtu) facies that are generated using the make_structured_mesh.py script from the SPE11 Git repository.')
    parser.add_argument('-c', '--case', default='.',
                        help='Identification of the SPE11 benchmark case: spe11a, spe11b or spe11c.')
    parser.add_argument('-f', '--filepath', default='.',
                        help='Path to the .vtu file generated by the make_structured_mesh.py script')

    args = parser.parse_args()

    filepath = args.filepath
    filename = os.path.splitext(os.path.basename(filepath))[0]
    work_dir = os.path.dirname(filepath)

    print(f' Msg: Input mesh path: {filepath}')
    print(f' Msg: Work dir: {work_dir}')

    case_name = args.case
    case_config = CASE_CONFIG[case_name]

    print(f' Msg: Case name: {case_name}')

    mesh = read_mesh_file(filepath, case_config['nI'])

    report_dir = os.path.join(work_dir, str(case_name))
    os.makedirs(report_dir, exist_ok=True)

    generate_region_index(mesh, case_config['nI'], filename, report_dir, 'SATNUM')
    #generate_region_index(mesh, CASE_CONFIG[case_name]['nI'], filename, report_dir, 'FIPNUM')
    generate_petro_param(mesh, case_name,  case_config['nI'], filename, report_dir, 'PORO')
    generate_petro_param(mesh, case_name,  case_config['nI'], filename, report_dir, 'PERMX')
    generate_petro_param(mesh, case_name,  case_config['nI'], filename, report_dir, 'THCONR')
    print(' ===========================================================================================\n')

if __name__ == '__main__':
    main()