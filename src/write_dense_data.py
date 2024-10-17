"""

__author__ = "fgasa"

"""
import re
import os
import argparse
import numpy as np
from opm.io.ecl import EclFile, ERst, EGrid, ESmry

REFERENCE_WATER_DENSITY = 998.107723
REFERENCE_GAS_DENSITY = 1.868433
BARS_TO_PASCALS = 1e5
GAS_MW = 44.01   #The molar mass of CO₂ is 44.01 g/mol or 0.04401 kg/mol'
CASE_CONFIG = {
    'spe11a': {
        'adjust_data_for_report': False,
        'time_interval': np.arange(0, 121, 1),
        #'time_to_remove': np.arange(2, 11, 2)#[2, 4, 6, 8, 10]
        'time_to_remove': [2, 4, 6, 8, 10]
    },
    'spe11b': {
        'adjust_data_for_report': True,
        'time_interval': np.arange(0, 1001, 5)
    },
    'spe11c': {
        'adjust_data_for_report': True,
        'time_interval': np.arange(0, 1001, 5),
        #'time_to_remove':np.arange(55, 1000, 5),
        'time_to_remove': [12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37,
                           38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63,
                           64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88,
                           89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                           111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129,
                           130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148,
                           149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167,
                           168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186,
                           187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200],
        'time_to_report': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                           600, 700, 800, 900, 1000]

    }
}

def convert_win_path_to_wsl(windows_path):
    windows_path = windows_path.replace('\\', '/')
    wsl_path = '/mnt/' + windows_path.replace(':', '')
    wsl_path = wsl_path.replace('\\', '/')
    wsl_path = wsl_path.lower()
    return wsl_path

def get_file_with_extension(folder_path, extension):
    for file in os.listdir(folder_path):
        if file.endswith('.' + extension):
            file_path = os.path.join(folder_path, file)

            return file_path

    # Return None if no file with the specified extension is found
    return None

def get_path_with_regex(directory, regex_pattern):
    found_files = []
    pattern = re.compile(regex_pattern)
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if pattern.match(filename):
                found_files.append(os.path.join(dirpath, filename))
    return found_files

# get model xyz for cell/element
def get_model_xyz(target_egrid, filtered_element):
    # target_egrid(<class 'opm.opmcommon_python.EGrid'>) and
    x_list, y_list, z_list = [], [], []
    for i in filtered_element:
        # retrieve i, j, k indices from the global index
        i, j, k = target_egrid.ijk_from_global_index(i) #<class 'list'>
        # get the XYZ coordinates from the i, j, k indices
        temp_xyz = target_egrid.xyz_from_ijk(i, j, k)#<class 'tuple'>

        # Unpack the coordinates and append to the respective lists
        x, y, z = temp_xyz

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    # Adjust xyz, calculate the average values from 8 nodes
    x_np_mean = np.mean(np.array(x_list), axis=1)
    y_np_mean = np.mean(np.array(y_list), axis=1)
    z_np_mean = np.mean(np.array(z_list), axis=1)

    return x_np_mean, y_np_mean, z_np_mean

# generate model inj indexes
def get_model_ijk(nI, nJ, nK):
    temp_i = np.arange(1, nI + 1, 1)
    temp_j = np.arange(1, nJ + 1, 1)
    temp_k = np.arange(1, nK + 1, 1)

    #index_i = np.tile(temp_i, nK) # this dublicate everything
    index_i = np.tile(temp_i, nK * nJ)
    index_j = np.tile(temp_j, nI * nK)
    index_k = np.repeat(temp_k, nI * nJ)

    return index_i, index_j, index_k

# get target dynamic array from rst
def get_array_from_rst(all_rst_paths, key_name):
    array_list = []
    for rst in all_rst_paths:
        rst_file = ERst(rst)
        temp_report_step = rst_file.report_steps[0]
        array_list.append(rst_file[key_name, temp_report_step])

    return np.array(array_list)

def main():
    
    write_dense_data = True

    parser = argparse.ArgumentParser(description='This script generates sparse data/map from OPM Flow'
                                                 ' simulator for CSP SEP11.')

    parser.add_argument('-c', '--case', default='.',
                        help='Identification of the SPE11 benchmark case: spe11a, spe11b or spe11c.')

    parser.add_argument('-f', '--folder', default='.',
                        help='Path to the folder containing the simulation files for genering reports.')

    args = parser.parse_args()

    working_dir = args.folder
    # double check this for operating on wsl
    working_dir_wsl = convert_win_path_to_wsl(working_dir)

    case_name = args.case
    case_config = CASE_CONFIG[case_name]
    print(' \n===========================================================================================')
    print(' Msg: The working dir is ', working_dir_wsl)

    egrid_path = get_file_with_extension(working_dir_wsl, 'EGRID')
    model_name = os.path.splitext(os.path.basename(egrid_path))[0]
    init_path = get_file_with_extension(working_dir_wsl, 'INIT')

    target_egrid = EGrid(egrid_path)
    init_file = EclFile(init_path)
    # ecl_egrid = EclGrid(egrid_path)

    unrst_pattern = r'.*\.X\d{4}$'  # Matches files ending with .X followed by exactly four digits
    all_rst_paths = get_path_with_regex(working_dir_wsl, unrst_pattern)

    if case_name=='spe11a':
        timesteps_to_remove = list(case_config['time_to_remove'])
        # might be index shifting: todo
        # sort the indices in reverse order to avoid index shifting timesteps_to_remove.sort(reverse=True)

        for index in timesteps_to_remove:
            del all_rst_paths[index]

    if case_name=='spe11c':
        # Convert timesteps to remove to a list of integers
        timesteps_to_remove = list(case_config['time_to_remove'])
        temp_time = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 121, 141, 161, 181, 200] # todo
        all_rst_paths = [path for i, path in enumerate(all_rst_paths) if
                         i not in timesteps_to_remove or i in temp_time]

    prefix = case_name + '_spatial_map_'
    report_dir = os.path.join(working_dir_wsl, case_name)
    os.makedirs(report_dir, exist_ok=True)

    nI, nJ, nK = target_egrid.dimension
    print(' Msg: Grid dimension(nI,nJ,nK): ', nI, nJ, nK)
    tot_ant_cells = nI * nJ * nK
    print(' Msg: Total elements in model: ', tot_ant_cells)
    actnum_info = target_egrid.active_cells
    print(' Msg: Total active elements in model: ', actnum_info)

    porevolume = np.array(init_file['PORV'])  # all elements in model 4253760
    actind = list(i for i, porv in enumerate(porevolume) if porv > 0)

    # Convert the generator to a 1D numpy array
    actind_new = np.fromiter(actind, dtype=int)

    # Convert to a list or a numpy array if needed
    filtered_element_list = list(actind_new)

    x_np_mean, y_np_mean, z_np_mean = get_model_xyz(target_egrid,filtered_element_list )

    index_i, index_j, index_k = get_model_ijk(nI, nJ, nK)

    if case_config['adjust_data_for_report']:

        east_boundary = np.where(index_i == 1)
        west_boundary = np.where(index_i == nI)
        east_boundary_nnc = np.where(index_i == 2)
        west_boundary_nnc = np.where(index_i == nI-1)

        # Combine the indices using the numpy.concatenate() function
        boundary_elements = np.concatenate([east_boundary, west_boundary])
        boundary_elements = boundary_elements.ravel()
        print(' Msg: Boundary elements are found')

        west_boundary_nnc = np.concatenate([west_boundary_nnc, west_boundary])
        east_boundary_nnc = np.concatenate([east_boundary, east_boundary_nnc])
        # west_boundary_nnc_flat = west_boundary_nnc.ravel()  # Flatten the array into a 1D array
        west_boundary_nnc = west_boundary_nnc.T  # 840    841] or [  1682   1683]
        east_boundary_nnc = east_boundary_nnc.T
        print(' Msg: Boundary elements and nnc are found')

        # update the last element+1 and first element+1 values. to mean or sum.. and then replace initial element
        def update_boundary_elements(target_np, east_boundary_nnc, west_boundary_nnc, method='average'):
            extracted_elements = target_np[east_boundary_nnc]
            pairs_of_elements = extracted_elements.reshape((-1, 2))
            # Compute the average of each pair of consecutive elements
            if method == 'average':
                updated_array_of_pairs = np.mean(pairs_of_elements, axis=1)  # all average information of two elements
            if method == 'sum':
                updated_array_of_pairs = np.sum(pairs_of_elements, axis=1)  # all average information of two elements

            east_nnc = east_boundary_nnc[:, 1]
            np.put(target_np, east_nnc, updated_array_of_pairs)

            extracted_elements = target_np[west_boundary_nnc]
            pairs_of_elements = extracted_elements.reshape((-1, 2))
            # Compute the average of each pair of consecutive elements
            if method == 'average':
                updated_array_of_pairs = np.mean(pairs_of_elements, axis=1)
            if method == 'sum':
                updated_array_of_pairs = np.sum(pairs_of_elements, axis=1)
            east_nnc = east_boundary_nnc[:, 1]

            west_nnc = west_boundary_nnc[:, 0]  # perfect, this this first column
            np.put(target_np, west_nnc, updated_array_of_pairs)

        # replace nnc to average of boundary and nnc element value
        update_boundary_elements(x_np_mean, east_boundary_nnc, west_boundary_nnc, method='average')

        # ---------------------------------------------------------------- remove boundary elements
        # Remove the elements from target array that have their indices in the boundary_element array

        x_np_mean = np.delete(x_np_mean, boundary_elements)  # delete to make shape for reporting
        z_np_mean = np.delete(z_np_mean, boundary_elements)

        # ---------------------------------------------------------------- we downscale static arrays
        if case_name=='spe11b':
            #x_report = np.mean(x_np_mean.reshape(-1, 2), axis=1)  # old method but correct in terms of size (100800,)
            #x_report = x_np_mean
            #y_report = y_np_mean
            x_report = np.mean(x_np_mean.reshape(-1, 2), axis=1)
            y_report = np.mean(y_np_mean.reshape(-1, 2), axis=1)
            z_report = np.mean(z_np_mean.reshape(-1, 2), axis=1)

        if case_name=='spe11c':
            x_report = x_np_mean
            z_report = z_np_mean

            y_np_mean = np.delete(y_np_mean, boundary_elements)
            y_report = y_np_mean
        print(' Msg: Initial simulation dimension is adjusted for reporting')
    else:
        x_report = x_np_mean
        y_report = y_np_mean
        z_report = z_np_mean
        print(' Msg: Initial simulation dimension is used for reporting')

    print(' Msg: New report shape for x, y, z report values are', x_report.shape, y_report.shape, z_report.shape)

    # to invert z for report
    z_report = np.flipud(z_report)

    sgas = get_array_from_rst(all_rst_paths, 'SGAS')
    gas_dens = get_array_from_rst(all_rst_paths, 'GAS_DEN')

    if case_name == 'spe11a':
        pcgw = get_array_from_rst(all_rst_paths, 'PCGW')
        pressure = get_array_from_rst(all_rst_paths, 'PRESSURE')
        pressure = (pressure - pcgw) * BARS_TO_PASCALS

        oil_dens = get_array_from_rst(all_rst_paths, 'WAT_DEN')
        rs = get_array_from_rst(all_rst_paths, 'RSW')
        rv = get_array_from_rst(all_rst_paths, 'RVW')
    else:
        pressure = get_array_from_rst(all_rst_paths, 'PRESSURE')
        pressure = pressure * BARS_TO_PASCALS
        temperature = get_array_from_rst(all_rst_paths, 'TEMP')
        oil_dens = get_array_from_rst(all_rst_paths, 'OIL_DEN')
        rs = get_array_from_rst(all_rst_paths, 'RS')
        rv = get_array_from_rst(all_rst_paths, 'RV')

    print(' Msg: All main arrays are derived from OPM rst')

    soil = 1 - sgas
    sgcr = 1.000000e-1
    # avoid strange saturation vales below -0.0001
    sgas[sgas <= 0] = 0
    gas_dens[sgas <= 0] = 0

    # from doi.org/10.1016/S1750-5836(07)00010-2
    x_gas = rs * REFERENCE_GAS_DENSITY / (REFERENCE_WATER_DENSITY + (rs * REFERENCE_GAS_DENSITY))
    x_oil = rv * REFERENCE_WATER_DENSITY / (REFERENCE_GAS_DENSITY + (rv * REFERENCE_WATER_DENSITY))

    gas_mass = porevolume * gas_dens * (1 - x_oil)

    imm_gas_spe = gas_mass / GAS_MW * sgas * (sgcr >= sgas)
    mob_gas_spe = gas_mass / GAS_MW * sgas * (sgas > sgcr)
    gas_dis = porevolume * soil * x_gas * oil_dens

    gas_gas = (imm_gas_spe + mob_gas_spe) * GAS_MW  # total gas doest work, since we don't want disolved fraction
    total_gas = gas_gas + gas_dis

    # the number of sim timesteps
    num_rows = sgas.shape[0]

    # this function, remap data from simulation grid to report grid, for example 5 m to 10m
    def generate_report_array(np_array, boundary_elements, east_boundary_nnc, west_boundary_nnc, method, case_name):
        # Update the boundary elements: first take average of two boudnary + nnc elements
        update_boundary_elements(np_array, east_boundary_nnc, west_boundary_nnc, method)
        # Remove the elements from target array that have their indices in the boundary_element array. this is element 1 and 842
        # here we reshape data in z direction!
        temp_report = np.delete(np_array, boundary_elements)
        if case_name == 'spe11b': #todo  if case_name == 'spe11b' and method in ['average', 'sum']: temp_report = getattr(np, method)(report_data.reshape(-1, 2), axis=1)
            if method == 'average':
                temp_report = np.mean(temp_report.reshape(-1, 2), axis=1)
            if method == 'sum':
                temp_report = np.sum(temp_report.reshape(-1, 2), axis=1)
        return temp_report

    if write_dense_data == True:
        for i in range(num_rows):
            report_time =case_config['time_interval'][i]
            if case_config['adjust_data_for_report']:
                # delete the first elements!  # Remove the elements from target array that have their indices in the boundary_element array
                pressure_report = generate_report_array(pressure[i, :], boundary_elements, east_boundary_nnc,
                                                        west_boundary_nnc, 'average',case_name)
                sgas_report = generate_report_array(sgas[i, :], boundary_elements, east_boundary_nnc, west_boundary_nnc,
                                                    'average', case_name)
                x_gas_report = generate_report_array(x_gas[i, :], boundary_elements, east_boundary_nnc,
                                                     west_boundary_nnc, 'average',case_name)
                x_oil_report = generate_report_array(x_oil[i, :], boundary_elements, east_boundary_nnc, west_boundary_nnc,
                                                     'average',case_name)
                gas_dens_report = generate_report_array(gas_dens[i, :], boundary_elements, east_boundary_nnc,
                                                        west_boundary_nnc, 'average',case_name)
                oil_dens_report = generate_report_array(oil_dens[i, :], boundary_elements, east_boundary_nnc,
                                                        west_boundary_nnc, 'average',case_name)
                total_gas_report = generate_report_array(total_gas[i, :], boundary_elements, east_boundary_nnc,
                                                         west_boundary_nnc,'sum', case_name)
                temperature_report = generate_report_array(temperature[i, :], boundary_elements, east_boundary_nnc,
                                                           west_boundary_nnc, 'average',case_name)
            else:
                pressure_report = pressure[i, :]
                sgas_report = sgas[i, :]
                x_gas_report = x_gas[i, :]
                x_oil_report = x_oil[i, :]
                gas_dens_report = gas_dens[i, :]
                oil_dens_report = oil_dens[i, :]
                total_gas_report = total_gas[i, :]
            # generate the report data and save it to a file
            if case_name=='spe11a':
                raw_data = np.column_stack((
                                           x_report, z_report, pressure_report, sgas_report, x_gas_report, x_oil_report,
                                           gas_dens_report, oil_dens_report, total_gas_report))
                all_headers = ['x [m]', 'z [m]', 'pressure [Pa]', 'gas saturation [-]',
                               'mass fraction of CO2 in liquid [-]', 'mass fraction of H20 in vapor [-]',
                               'phase mass density gas [kg/m3]', 'phase mass density water [kg/m3]',
                               'total mass CO2 [kg]']
                report_name = f'{prefix}{report_time}h.csv'
            elif case_name == 'spe11b':
                raw_data = np.column_stack((
                                           x_report, z_report, pressure_report, sgas_report, x_gas_report, x_oil_report,
                                           gas_dens_report, oil_dens_report, total_gas_report, temperature_report))
                all_headers = ['x [m]', 'z [m]',
                               'pressure [Pa]', 'gas saturation [-]',
                               'mass fraction of CO2 in liquid [-]', 'mass fraction of H20 in vapor [-]',
                               'phase mass density gas [kg/m3]', 'phase mass density water [kg/m3]',
                               'total mass CO2 [kg]', 'temperature [C]']
                report_name = f'{prefix}{report_time}y.csv'
            elif case_name == 'spe11c':
                raw_data = np.column_stack((x_report, y_report, z_report, pressure_report, sgas_report, x_gas_report,
                                            x_oil_report, gas_dens_report, oil_dens_report, total_gas_report,
                                            temperature_report))
                all_headers = ['x [m]', 'y [m]', 'z [m]', 'pressure [Pa]', 'gas saturation [-]',
                               'mass fraction of CO2 in liquid [-]', 'mass fraction of H20 in vapor [-]',
                               'phase mass density gas [kg/m3]', 'phase mass density water [kg/m3]',
                               'total mass CO2 [kg]', 'temperature [C]']
                report_time = case_config['time_to_report'][i]
                report_name = f'{prefix}{report_time}y.csv'

            header_string = ','.join(all_headers)
            output_path = '{}/' + str(report_name)
            np.savetxt(output_path.format(report_dir), raw_data, header=header_string, delimiter=',', fmt='%.3e')
            print(f' Msg: Saved {report_name}')

    print(' ===========================================================================================\n')

if __name__ == '__main__':
    main()