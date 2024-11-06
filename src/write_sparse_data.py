'''

__author__ = 'fgasa'

'''
import os
import re
import argparse
import datetime
import numpy as np
from opm.io.ecl import EclFile, ERst, EGrid, ESmry

REFERENCE_WATER_DENSITY = 998.107723
REFERENCE_GAS_DENSITY = 1.868433
GAS_MW = 44.01   #The molar mass of CO₂ is 44.01 g/mol or 0.04401 kg/mol
BARS_TO_PASCALS = 1e5
CASE_CONFIG = {
    'spe11a': {
        'report_columns': 10,
        'init_time': 1800,
        'end_time': 4.32e5,
        'report_time_interval': 600,
        'pressure1': 'BPR:150,1,70',
        'pressure2': 'BPR:170,1,10',
        'all_headers': ['t [s]', 'p1 [Pa]', 'p2 [Pa]', 'mobA [kg]', 'immA [kg]', 'dissA [kg]', 'sealA [kg]', 'mobB [kg]', 'immB [kg]', 'dissB [kg]', 'sealB [kg]', 'M_C [m2]', 'sealTot [kg]']
    },
    'spe11b': {
        'report_columns': 11,
        'init_time': 3.1536e10,
        'end_time': 3.1536e10,
        'report_time_interval': 3.1536e6,
        'pressure1': 'BPR:451,1,140',
        'pressure2': 'BPR:511,1,20',
        'all_headers': ['t [s]', 'p1 [Pa]', 'p2 [Pa]', 'mobA [kg]', 'immA [kg]', 'dissA [kg]', 'sealA [kg]', 'mobB [kg]', 'immB [kg]', 'dissB [kg]', 'sealB [kg]', 'M_C [m2]', 'sealTot [kg]']
    },
    'spe11c': {
        'report_columns': 11,
        'init_time': 3.1536e10,
        'end_time': 3.1536e10,
        'report_time_interval': 3.1536e6,
        'pressure1': 'BPR:90,50,70',
        'pressure2': 'BPR:102,50,10',
        'all_headers': ['t [s]', 'p1 [Pa]', 'p2 [Pa]', 'mobA [kg]', 'immA [kg]', 'dissA [kg]', 'sealA [kg]', 'mobB [kg]', 'immB [kg]', 'dissB [kg]', 'sealB [kg]', 'M_C [m2]', 'sealTot [kg]', 'boundaryTot [kg]']
    }
}

def get_path_with_extension(folder_path, extension):
    for file in os.listdir(folder_path):
        if file.endswith('.' + extension):
            file_path = os.path.join(folder_path, file)
            return file_path

    # Return None if no file with the specified extension is found
    return None

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
    return None

def get_path_with_regex(directory, regex_pattern):
    found_files = []
    pattern = re.compile(regex_pattern)
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if pattern.match(filename):
                found_files.append(os.path.join(dirpath, filename))
    return found_files

def get_array_from_rst(all_rst_paths, key_name):
    array_list = []
    for rst in all_rst_paths:
        rst_file = ERst(rst)
        temp_report_step = rst_file.report_steps[0]
        array_list.append(rst_file[key_name, temp_report_step])

    return np.array(array_list)

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

def generate_times_series_smspec(smspec_path, pressure1, pressure2,case_name):
    summary = ESmry(smspec_path)
    days = summary['TIME']
    seconds = days * 24*60*60 # Day to sec
    if case_name =='spe11a':
        pop1 = (summary[pressure1] - 0.014999) * BARS_TO_PASCALS #todo
        pop2 = (summary[pressure2] - 0.014999) * BARS_TO_PASCALS
    else:
        pop1 = summary[pressure1] * BARS_TO_PASCALS
        pop2 = summary[pressure2] * BARS_TO_PASCALS

    #mc = summary * 0    #mc = np.full_like(mc, np.nan)  # 'n/a'
    raw_data = np.column_stack((seconds, pop1, pop2))
    print(f' Msg: The raw data from SMSPEC file is derived')

    return raw_data

def resample_data(data, time_init, end_time, use_start_index, time_interval):
    '''Resample time series data either from a specific start index or from the beginning of simulation'''
    if use_start_index:
        start_index = np.where(data[:, 0] >= time_init)[0][0]
        time = data[start_index:, 0] - time_init
        other_columns = data[start_index:, 1:]
    else:
        time = data[:, 0] #use entire dataset,  starting from time 0
        other_columns = data[:, 1:]

    # new time grid
    new_time_grid = np.arange(0, end_time + time_interval, time_interval) # 4.32e5 for spe11a

    # Interpolate other columns onto the new time grid
    resampled_data = np.zeros((len(new_time_grid), other_columns.shape[1]))
    for i in range(other_columns.shape[1]):
        resampled_data[:, i] = np.interp(new_time_grid, time, other_columns[:, i])

    print(f' Msg: Resampled data is generated')


    if use_start_index:
        # combine resampled time with other columns when using start index
        resampled_array = np.column_stack((new_time_grid, resampled_data))
        return resampled_array
    else:
        return resampled_data

def generate_times_series_rst(output_dir, init_path, egrid_path, case_name):
    regex_pattern = r'.*\.X\d{4}$'
    all_rst_paths = get_path_with_regex(output_dir, regex_pattern)

    target_egrid = EGrid(egrid_path)
    init_file = EclFile(init_path)

    nI, nJ, nK = target_egrid.dimension
    print(f' Msg: Grid dimension(nI,nJ,nK): {nI}, {nJ}, {nK}')

    tot_ant_cells = nI * nJ * nK
    print(f' Msg: Total elements in model: {tot_ant_cells}')
    actnum_info = target_egrid.active_cells
    print(f' Msg: Total active elements in model: {actnum_info}')

    porevolume = np.array(init_file['PORV'])  # all elements in model 4253760
    fipnum = np.array(init_file['FIPNUM'])
    satnum = np.array(init_file['SATNUM'])

    actind = list(i for i, porv in enumerate(porevolume) if porv > 0)
    actind_new = np.fromiter(actind, dtype=int) #todo
    filtered_element_list = list(actind_new)
    x_coord, y_coord, z_coord = get_model_xyz(target_egrid, filtered_element_list)

    sgas = get_array_from_rst(all_rst_paths, 'SGAS')
    gas_dens = get_array_from_rst(all_rst_paths, 'GAS_DEN')

    # Assuming the inputs are numpy arrays
    num_time_steps, num_locations = sgas.shape
    #print(x_coord)
    if case_name == 'spe11a':
        pcgw = get_array_from_rst(all_rst_paths, 'PCGW')
        pressure = get_array_from_rst(all_rst_paths, 'PRESSURE')
        pressure = (pressure - pcgw) * BARS_TO_PASCALS

        oil_dens = get_array_from_rst(all_rst_paths, 'WAT_DEN')
        rs = get_array_from_rst(all_rst_paths, 'RSW')
        rv = get_array_from_rst(all_rst_paths, 'RVW')

        time_interval = np.arange(0, num_time_steps, 1)  # keep in mind first 5 hours has timestep size 30 m
        #time_interval = time_interval.reshape(121, 1) * 3600
        time_interval = time_interval.reshape(num_time_steps, 1) * 3600

        # Box A: bottom left (1.1, 0.0), top right (2.8, 0.6)
        # Box B: bottom left (0.0, 0.6), top right (1.1, 1.2)
        # Box C: bottom left (1.1, 0.1), top right (2.6, 0.4)

        # Create boolean masks that are independent of the time series
        mask_box_a = ((x_coord >=1.10) & (x_coord < 2.8)) | (z_coord > 0.6)
        mask_box_b = (x_coord >=0) | (x_coord < 1.1) | (z_coord <= 0.6)
        mask_seal_a = mask_box_a & (satnum == 1)
        mask_seal_b = mask_box_b & (satnum == 1)
        mask_seal = (satnum == 1)# | ((fipnum >= 1) & (fipnum <= 3))

    else:
        pressure = get_array_from_rst(all_rst_paths, 'PRESSURE')
        pressure = pressure * BARS_TO_PASCALS

        oil_dens = get_array_from_rst(all_rst_paths, 'OIL_DEN')
        rs = get_array_from_rst(all_rst_paths, 'RS')
        rv = get_array_from_rst(all_rst_paths, 'RV')

        time_interval = np.arange(0, 1001, 5)
        time_interval = time_interval.reshape(201, 1) * 3.154e+7

        # BOX A: bottom left (3300, 0), top right (8300, 600)
        # Box B: bottom left (100, 600), top right (3300, 1200)
        # Box C: bottom left (3300, 100), top right (7800, 400)

        mask_box_a = ((x_coord >= 3300) & (x_coord < 8300)) & (z_coord > 600)
        mask_box_b = (x_coord > 100) & (x_coord < 3300) & (z_coord <= 600)
        mask_seal_a = mask_box_a & (satnum == 1)
        mask_seal_b = mask_box_b & (satnum == 1)
        mask_seal = (satnum == 1)

        # CO2 mass in all boundary volumes (CO2 in any form within the region where LB0 !=0 )
        mask_boundary = (fipnum == 12) #to do, ((x_coord >= 8399)&(z_coord < 676)&(z_coord > 820)) & ((x_coord <= 1) & (z_coord < 744) & (z_coord > 895) ]

    print(f' Msg: Main arrays are derived from OPM rst')

    soil = 1 - sgas
    sgcr = 1.000000e-1
    # avoid strange saturation vales below -0.0001
    sgas[sgas < 0] = 0
    gas_dens[sgas <= 0] = 0

    # Option from doi.org/10.1016/S1750-5836(07)00010-2
    x_gas = rs * REFERENCE_GAS_DENSITY / (REFERENCE_WATER_DENSITY + (rs * REFERENCE_GAS_DENSITY))
    x_oil = rv * REFERENCE_WATER_DENSITY / (REFERENCE_GAS_DENSITY + (rv * REFERENCE_WATER_DENSITY))

    gas_mass = porevolume * gas_dens * (1 - x_oil)

    imm_gas_spe = gas_mass / GAS_MW * sgas * (sgcr >= sgas)
    mob_gas_spe = gas_mass / GAS_MW * sgas * (sgas > sgcr)
    gas_dis = porevolume * soil * x_gas * oil_dens

    gas_gas = (imm_gas_spe + mob_gas_spe) * GAS_MW  # total gas doest work, since we dont want disolved fraction
    #total_gas = gas_mass + gas_dis #
    total_gas = gas_gas + gas_dis

    print(f' Msg: Reporting arrays are generated')

    results = np.zeros((num_time_steps, CASE_CONFIG[case_name]['report_columns']))

    # Loop through each time step in the time series
    for i in range(num_time_steps):
        sgas_report = sgas[i, :]
        gas_gas_report = gas_gas[i, :] #gas_mass is almost correct, but kiel potentially used wrong evaluation of immobile CO2.
        total_gas_report = total_gas[i, :]
        gas_dis_report = gas_dis[i, :]

        # Adjust masks for fluid saturation
        mask_mob_a = mask_box_a & (sgas_report > sgcr)
        mask_imm_a = mask_box_a & (sgas_report <= sgcr)
        mask_mob_b = mask_box_b & (sgas_report > sgcr)
        mask_imm_b = mask_box_b & (sgas_report <= sgcr)

        # Get primary report outputs
        results[i, 0] = np.sum(gas_gas_report[mask_mob_a])
        results[i, 1] = np.sum(gas_gas_report[mask_imm_a])
        results[i, 2] = np.sum(gas_dis_report[mask_box_a])
        results[i, 3] = np.sum(total_gas_report[mask_seal_a])

        results[i, 4] = np.sum(gas_gas_report[mask_mob_b])
        results[i, 5] = np.sum(gas_gas_report[mask_imm_b])
        results[i, 6] = np.sum(gas_dis_report[mask_box_b])
        results[i, 7] = np.sum(total_gas_report[mask_seal_b])
        results[i, 8] = np.nan #   #'n/a'  mc
        results[i, 9] = np.sum(total_gas_report[mask_seal])  # sealTot

    results = np.hstack((time_interval, results))  # include time column

    return results

def write_time_series(array, header, output_dir, case_name, perfix):
    report_name = case_name + '_time_series' + perfix + '.csv'
    output_path = '{}/' + str(report_name)
    np.savetxt(output_path.format(output_dir), array, delimiter=',', header=header, fmt='%.4e', comments='')
    print(' Msg: Time series is written successfully ', report_name)

def main():
    start = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='This script generates performance data report from OPM Flow/ECLIPSE'
                                                 'reservoir simulator for CSP SEP11.')
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
    print(f' Start: {start}')
    print(f' Msg: Working folder: {working_dir_wsl}')

    report_dir = os.path.join(working_dir_wsl, case_name)
    os.makedirs(report_dir, exist_ok=True)
    print(f' Msg: Report folder: {report_dir}')

    init_path = get_file_with_extension(working_dir_wsl, 'INIT')
    egrid_path = get_file_with_extension(working_dir_wsl, 'EGRID')
    smspec_path = get_path_with_extension(working_dir_wsl, 'SMSPEC')

    results_part1 = generate_times_series_smspec(smspec_path, case_config['pressure1'], case_config['pressure2'], case_name)
    results_part2 = generate_times_series_rst(working_dir_wsl, init_path, egrid_path, case_name)

    all_headers = ','.join(case_config['all_headers'])

    #write_time_series(array_detailed, all_headers, report_folder, case_name, '_detailed')

    results_part1 = resample_data(results_part1,  case_config['init_time'], case_config['end_time'], True, case_config['report_time_interval'])

    results_part2 = resample_data(results_part2,case_config['init_time'], case_config['end_time'], False, case_config['report_time_interval'])

    array_resampled = np.hstack((results_part1, results_part2))

    write_time_series(array_resampled, all_headers, report_dir, case_name, '')

    end = datetime.datetime.now()
    print(f' End: {end}')
    print(' ===========================================================================================\n')

if __name__ == '__main__':
    main()