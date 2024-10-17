"""

__author__ = "fgasa"

"""
import os
import argparse
import numpy as np
from opm.io.ecl import EGrid, ESmry

REFERENCE_GAS_DENSITY = 1.868433
CASE_CONFIG = {
    'spe11a': {
        'init_time': 1800,
        'end_time': 4.32e5,
        'report_time_interval': 600,
        'dof': 2
    },
    'spe11b': {
        'init_time': 3.1536e+10,
        'end_time': 3.1536e+10,
        'report_time_interval': 3.1536e6,
        'dof': 3
    }
}

'''
Here is the description of what the simulator provides as additional convergence output files (from SimulatorReport::reportFullyImplicit).
https://github.com/OPM/opm-simulators/blob/3f6642ee13633a863c324829f6abbcabb5877633/opm/simulators/flow/SimulatorFullyImplicitBlackoil.hpp#L63
https://github.com/OPM/opm-simulators/blob/master/opm/simulators/timestepping/SimulatorReport.cpp
'''

def get_path_with_extension(folder_path, extension):
    for file in os.listdir(folder_path):
        if file.endswith('.' + extension):
            file_path = os.path.join(folder_path, file)
            return file_path
    return None # Return None if no file with the specified extension is found

def convert_win_path_to_wsl(windows_path):
    windows_path = windows_path.replace('\\', '/')
    wsl_path = '/mnt/' + windows_path.replace(':', '')
    wsl_path = wsl_path.replace('\\', '/')
    wsl_path = wsl_path.lower()
    return wsl_path

def write_time_series(np_array, header, output_dir, case_name, perfix):
    report_name = case_name + '_performance_time_series' + perfix + '.csv'
    output_path = '{}/' + str(report_name)
    np.savetxt(output_path.format(output_dir), np_array, delimiter=',', header=header, fmt='%.4e', comments='')
    print(' Msg: Time series is written successfully ', report_name)

def generate_performance_data(smspec_path, simlog_path, active_elements, degrees_of_freedom):
    """ Generate performance data from SMSPEC and INFOSTEP files"""

    degrees_of_freedom = float(active_elements * degrees_of_freedom)
    summary = ESmry(smspec_path)

    # mass based on FGIT leads to  deviations in the dissolved, mobile and immobile fractions later (-> use FGIP)
    fgip = summary['FGIP']
    fgip_mass = fgip.astype(float) * REFERENCE_GAS_DENSITY
    time_smry = summary['TIME']

    runtime = summary['TCPU']  # in seconds
    print(f' Msg: SMSPEC shape {len(time_smry)}')

    with open(simlog_path, 'r') as file:
        data = file.readlines()

    # extract only the relevant information to 'list'
    values = [line.split() for line in data[1:]]

    time = np.array([float(row[0]) * 86400 for row in values])
    # some time is Conv!=0, therefore time is 0
    tstep = np.array([float(row[1]) * float(row[11]) * 86400 for row in values])
    tstep_size = np.array([float(row[1]) * 86400 for row in values])
    time[-1] += tstep_size[-1]  # to fix last time step(63072e6 sec or 3.1536e+10)

    print(' Msg: INFOSTEP shape ', len(tstep))

    # initial column with tistem conv, where 1 is solved and 0 is failed
    fsteps = np.array([0 if x == 1 else (x + 1 if x == 0 else x) for x in [int(row[11]) for row in values]])
    dof = np.full_like(time, degrees_of_freedom)

    nliter = np.array([int(row[9]) for row in values])
    nres = np.array([float(row[8]) for row in values])
    # total number of nonlinear iterations for all timesteps = Newton iteration
    liniter = np.array([int(row[10]) for row in values])
    tlinsol = np.array([float(row[4]) for row in values])

    def make_simple_interp(a, b):
        a = np.asarray(a)
        b = np.asarray(b)

        # if the shapes already match, return a
        if a.shape == b.shape:
            return a

        # otherwise, Linearly interpolate a to match the shape of b
        return np.interp(np.linspace(0, 1, len(b)), np.linspace(0, 1, len(a)), a)

    # Linear interpolation using np.interp
    fgip_mass_interp = make_simple_interp(fgip_mass, time)
    runtime_interp = make_simple_interp(runtime, time)

    # fix it according to updates from the reporting routines as of August 2024
    # total runtime for advancing the solution in the last 600 s or 3.1536e6 s
    runtime_interp = np.diff(runtime_interp, prepend=runtime_interp[0])

    # before the injection phase (initialisation 1000 y or 30 min), the dof is equal to the mesh size
    dof_final = np.where(fgip_mass_interp == 0, active_elements, dof)

    all_data = np.column_stack((time, tstep, fsteps, fgip_mass_interp,
                                dof_final, nliter, nres,
                                liniter, runtime_interp, tlinsol))
    print(' Msg: Performance time series is generated')
    return all_data

def resample_data(data, time_init, end_time, use_start_index, time_interval):
    '''
    Resample time series data either from a specific start index or from the beginning of simulation.
    '''

    if use_start_index:
        # Find the start index where time is greater than or equal to time_init
        start_index = np.where(data[:, 0] >= time_init)[0][0]
        time = data[start_index:, 0] - time_init
        other_columns = data[start_index:, 1:]
    else:
        # Use entire dataset, starting from time 0
        time = data[:, 0]
        other_columns = data[:, 1:]

    # Determine new time grid
    new_time_grid = np.arange(0, end_time + time_interval, time_interval) # 4.32e5 for spe11

    # Interpolate other columns onto the new time grid
    resampled_data = np.zeros((len(new_time_grid), other_columns.shape[1]))
    for i in range(other_columns.shape[1]):
        resampled_data[:, i] = np.interp(new_time_grid, time, other_columns[:, i])

    print(' Msg: Resampled data is generated')

    if use_start_index:
        # Combine resampled time with other columns when using start index
        resampled_array = np.column_stack((new_time_grid, resampled_data))
        return resampled_array
    else:
        return resampled_data

def main():
    #start = datetime.datetime.now()
    #print(' Start: ', start)

    parser = argparse.ArgumentParser(description='This script generates performance data report from OPM Flow'
                                                 ' simulator for CSP SEP11.')
    parser.add_argument('-c', '--case', default='.',
                        help='Identification of the SPE11 benchmark case: spe11a, spe11b or spe11c.')

    parser.add_argument('-f', '--folder', default='.',
                        help='Path to the folder containing the simulation files for genering reports.')

    args = parser.parse_args()

    working_dir = args.folder
    # convert all Windows paths to WSL type, because OPM rst and OPM's python packages are linux-based
    working_dir_wsl = convert_win_path_to_wsl(working_dir)

    case_name = args.case
    case_config = CASE_CONFIG[case_name]

    report_dir = os.path.join(working_dir_wsl, str(case_name))
    os.makedirs(report_dir, exist_ok=True)
    print(' \n===========================================================================================')
    print(' Msg: Report folder ', report_dir)

    smspec_path = get_path_with_extension(working_dir_wsl, 'SMSPEC')
    print(' Msg: Simulation SMSPEC filename ', os.path.basename(smspec_path))

    simlog_path = get_path_with_extension(working_dir_wsl, 'INFOSTEP')
    print(' Msg: Convergence output filename ', os.path.basename(simlog_path))

    egrid_path = get_path_with_extension(working_dir_wsl, 'EGRID')
    target_egrid = EGrid(egrid_path)
    active_elements = target_egrid.active_cells

    raw_data = generate_performance_data(smspec_path, simlog_path, active_elements, case_config['dof'])

    all_headers = ['t [s]', 'tstep [s]', 'fsteps [-]', 'mass [kg]', 'dof [-]', 'nliter [-]', 'nres [-]', 'liniter [-]',
                   'runtime [s]', 'tlinsol [s]']
    all_headers = ','.join(all_headers)

    write_time_series(raw_data, all_headers, report_dir, case_name, '_detailed')

    perform_data = resample_data(raw_data, case_config['init_time'], case_config['end_time'], True, case_config['report_time_interval'])
    write_time_series(perform_data, all_headers, report_dir, case_name, '')

    print(' ===========================================================================================\n')

if __name__ == '__main__':
    main()