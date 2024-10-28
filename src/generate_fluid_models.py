"""

__author__ = "fgasa"

"""
import os
import numpy as np
import CoolProp.CoolProp as CP
import argparse
from scipy.special import erf

STD_TEMPERATURE = 15.56 + 273.15
STD_PRESSURE = 1.01325 * 1e5
REFERENCE_GAS = 'CO2'
REFERENCE_WATER = 'H2O'
BARS_TO_PASCALS = 1e5
REFERENCE_GAS_DENSITY = 1.868433
REFERENCE_WATER_DENSITY = 998.107723
GAS_MW = 44.01   #The molar mass of CO₂ is 44.01 g/mol or 0.04401 kg/mol
WATER_MW = 18.01

CP_VERSION = f"-- CoolProp version: {CP.get_global_param_string('version')}, gitrevision: {CP.get_global_param_string('gitrevision')}\n"
REFERENCE_CONDITION = f"-- Standard temperature: {STD_TEMPERATURE - 273.15} C, Standard pressure: {STD_PRESSURE / BARS_TO_PASCALS} bar\n"

CASE_CONFIG = {
    'spe11a': {
        'sw_imm': [0.32, 0.14, 0.12, 0.12, 0.12, 0.10, 0],
        'sg_imm': np.full(7, 0.1).tolist(),
        'pc_entry': (np.array([1500, 300, 100, 25, 10, 1, 1]) / BARS_TO_PASCALS).tolist(),
        'pc_max': np.full(7, 9.5e-1).tolist(),
        'c_a1': 2,
        'rela_data_points': 1000,
        'pvt_data_points': 500,
        'temperature': 20.0 + 273.15,
        'min_pressure': 1.0 * BARS_TO_PASCALS,
        'max_pressure': 6.0 * BARS_TO_PASCALS
    },
    'spe11b': {
        'sw_imm': [0.32, 0.14, 0.12, 0.12, 0.12, 0.10, 0],
        'sg_imm': np.full(7, 0.1).tolist(),
        'poro': [0.1, 0.2, 0.2, 0.2, 0.25, 0.35, 1e-6],
        'perm': [1e-16, 1e-13, 2e-13, 5e-13, 1e-12, 2e-12, 9.87e-21],
        'pc_entry': (np.sqrt(np.array([0.1, 0.2, 0.2, 0.2, 0.25, 0.35, 1e-6]) /
                      np.array([1e-16, 1e-13, 2e-13, 5e-13, 1e-12, 2e-12, 9.87e-21])) * 0.00612 / BARS_TO_PASCALS).tolist(),
        'pc_max': np.full(7, 300).tolist(),
        'c_a1': 1.5,
        'rela_data_points': 250,
        'pvt_data_points': 1000,
        'temperature': 55 + 273.15,
        'min_pressure': 150 * BARS_TO_PASCALS,
        'max_pressure': 550 * BARS_TO_PASCALS
    },
    'spe11c': {
        'sw_imm': [0.32, 0.14, 0.12, 0.12, 0.12, 0.10, 0],
        'sg_imm': np.full(7, 0.1).tolist(),
        'poro': [0.1, 0.2, 0.2, 0.2, 0.25, 0.35, 1e-6],
        'perm': [1e-16, 1e-13, 2e-13, 5e-13, 1e-12, 2e-12, 9.87e-21],
        'pc_entry': (np.sqrt(np.array([0.1, 0.2, 0.2, 0.2, 0.25, 0.35, 1e-6]) /
                      np.array([1e-16, 1e-13, 2e-13, 5e-13, 1e-12, 2e-12, 9.87e-21])) * 0.00612 / BARS_TO_PASCALS).tolist(),
        'pc_max': np.full(7, 300).tolist(),
        'c_a1': 1.5,
        'rela_data_points': 250,
        'pvt_data_points': 500,
        'temperature': 55 + 273.15,
        'min_pressure': 150 * BARS_TO_PASCALS,
        'max_pressure': 550 * BARS_TO_PASCALS
    }
}

def get_fluid_fugacity_coefficients(temperature, pressure, co2_density, fluid):
    '''Compute equilibrium constants and fugacity coefficients for CO2 and H2O, this is rewritten
    from github.com/Simulation-Benchmarks/11thSPE-CS11thSPE-CSP/thermodynamics'''
    TinC = temperature - 273.15
    R = 83.1446261815324
    p_bar = pressure / BARS_TO_PASCALS
    V = 1 / (co2_density / 44.01e-3) * 1e6  # molar volume

    if fluid == 'CO2':
        c = [1.189, 1.304e-2, -5.446e-5]  # CO2 constants
        logk0 = c[0] + c[1] * TinC + c[2] * TinC * TinC
        k0 = np.power(10, logk0)
        a = (7.54e7 - 4.13e4 * temperature)
        b = 27.8
    else:
        c = [-2.209, 3.097e-2, -1.098e-4, 2.048e-7]  # H2O constants
        logk0 = c[0] + c[1] * TinC + c[2] * TinC * TinC + c[3] * TinC ** 3
        k0 = np.power(10, logk0)
        a = 7.89e7
        b = 18.18

    lnPhi = (np.log(V / (V - b)) + b / (V - b)
             - 2 * a / (R * np.power(temperature, 1.5) * b) * np.log((V + b) / V)
             + a * b / (R * np.power(temperature, 1.5) * b ** 2) * (np.log((V + b) / V) - b / (V + b))
             - np.log(p_bar * V / (R * temperature)))
    phi = np.exp(lnPhi)

    return phi, k0

def generate_fluid_rela(case_name):
    ''' Generates relative permeability and saturation tables for all regions and writes them to one file. This function 
    has been primarily rewritten from github.com/Simulation-Benchmarks/11thSPE-CSP/blob/main/thermodynamics/writeFluidFlowerSGOF '''

    case_config = CASE_CONFIG[case_name]
    sw_imm, sg_imm, pc_entry, pc_max, c_a1, data_points = (
        case_config['sw_imm'], case_config['sg_imm'], case_config['pc_entry'],
        case_config['pc_max'], case_config['c_a1'], case_config['rela_data_points'])

    all_facies = []

    for facies in range(len(sw_imm)):
        sat = np.concatenate(([0], np.linspace( sg_imm[facies], 1, data_points - 1)))
        sg = np.maximum((sat - sg_imm[facies]) / (1 - sg_imm[facies]), 0)
        so = np.maximum((1 - sat - sw_imm[facies]) / (1 - sw_imm[facies]), 0)
        krg = sg ** c_a1
        kro = so ** c_a1

        so_adjusted = np.where(so == 0, np.finfo(float).eps, so)
        pc_og = pc_entry[facies] * so_adjusted ** (1 - c_a1)
        pc_og = pc_max[facies] * erf((pc_og / pc_entry[facies]) * (np.sqrt(np.pi) / 2))

        # Combining the results into one array for easier writing
        result_data = np.vstack((sat, krg, kro, pc_og)).T

        all_facies.append(result_data)
        all_facies.append(np.full((1, result_data.shape[1]), np.nan))  # Use NaNs for separation(/)

    all_facies = np.vstack(all_facies)
    return all_facies

def get_fluid_properties(temperature, min_pressure, max_pressure, sampling_points, fluid):
    pressures = np.linspace(min_pressure, max_pressure, sampling_points)
    properties = np.zeros((sampling_points, 8))
    for i, pressure in enumerate(pressures):
        properties[i, :2] = temperature, pressure
        properties[i, 2:8] = [CP.PropsSI(prop, 'T', temperature, 'P', pressure, fluid)
                              for prop in ['D', 'V', 'H', 'L', 'O', 'C']]
        # enthalpy [J/kg], this unit is always consistent from CoolProp
    
    return properties

def compute_A_B(temperature, pressure, co2_density, fluid):
    deltaP = pressure / BARS_TO_PASCALS - 1
    p_bar = pressure / BARS_TO_PASCALS

    R = 83.1446261815324
    v_av = 18.1 if fluid == 'H2O' else 32.6

    phi, k0 = get_fluid_fugacity_coefficients(temperature, pressure, co2_density, fluid)

    temp_var = (k0 / (phi * p_bar) * np.exp(deltaP * v_av / (R * temperature))) if fluid == 'H2O' else \
               (phi * p_bar / (55.508 * k0) * np.exp(-deltaP * v_av / (R * temperature)))
    return temp_var

def get_fluid_solubility(properties):
    temperature = properties[:, 0]
    pressure = properties[:, 1]
    co2_density = properties[:, 2]

    A = compute_A_B(temperature, pressure, co2_density, REFERENCE_WATER)
    B = compute_A_B(temperature, pressure, co2_density, REFERENCE_GAS)
    # calculate solubility
    y_H2O = (1 - B) / (1 / A - B)
    x_CO2 = B * (1 - y_H2O)
    data = np.column_stack((temperature, pressure, y_H2O, x_CO2))
    return data

def write_data(np_array, header, output_dir, fluid, perfix, usr_delimiter, case_name):
    filename = case_name + "_"  + perfix + "_" + fluid + '.txt'
    output_path = os.path.join(output_dir, filename)
    output_path = (output_path).upper()

    header = ','.join(header)

    np.savetxt(output_path, np_array, delimiter=usr_delimiter, header=header, fmt='%.6e', comments='')
    print(f' Msg: {output_path} is written successfully')

def write_gas_pvtx(pvt_array, solub_array, ref_gas_density, ref_oil_density, pvt_type, fluid, output_dir):
    filename = f"{pvt_type}_{fluid}_CP.INC"
    output_path = os.path.join(output_dir, filename).upper()

    pressure = pvt_array[:, 1] / BARS_TO_PASCALS
    temperature = pvt_array[0, 0]

    fluid_density = pvt_array[:, 2]
    fluid_viscosity = pvt_array[:, 3] * 1e3 #convert viscosity from Pa.s to cP
    fluid_fvf = ref_gas_density / fluid_density

    x_oil = solub_array[:, 2]
    x_gas = solub_array[:, 3]
    rv = (x_oil * WATER_MW * ref_gas_density) / ((1 - x_oil) * GAS_MW * ref_oil_density + x_oil * WATER_MW)  # mass ratio

    if pvt_type == 'PVTG':
        fluid_fvf = ref_gas_density / fluid_density / (1-x_gas)
        data = np.column_stack((pressure, rv, fluid_fvf, fluid_viscosity))
        with open(output_path, 'w') as f:
            f.write(CP_VERSION)
            f.write(REFERENCE_CONDITION)
            f.write('-- Pressure [bar] RV [m³/m³] Fluid FVF [rm³/sm³] Viscosity [cP]\nPVTG\n')
            for i in range(data.shape[0]):
                f.write(f"{pressure[i]:.6e} {rv[i]:.6e} {fluid_fvf[i]:.6e} {fluid_viscosity[i]:.6e}\n")
                f.write(f"\t\t\t {rv[i]*0:.6e} {fluid_fvf[i]:.6e} {fluid_viscosity[i]:.6e} /\n")
        print(f' Msg: Wet gas PVT with vaporized oil is generated (inc. oil vapor-gas ratio')
        return
    elif pvt_type == 'PVDG':
        header = ['-- Pressure [bar]', 'Fluid FVF [rm³/sm³]', 'Viscosity [cP]\nPVDG']
        data = np.column_stack((pressure, fluid_fvf, fluid_viscosity))
        print(f' Msg: Dry gas fluid PVT is generated')
    elif pvt_type == 'PVZG':
        zfactor = CP.PropsSI('Z', 'T', temperature, 'P', pressure * BARS_TO_PASCALS, fluid)
        header = ['-- Pressure [bar]', 'Compressibility Z-factor[-]', 'Viscosity [cP]\nPVZG\n'+str(temperature-273.15)+' /']
        data = np.column_stack((pressure, zfactor, fluid_viscosity))
        print(f' Msg: Dry gas fluid PVT with Z-factor is generated.')
    
    # Combine the CP_VERSION and REFERENCE_CONDITION with the PVT header
    header_lines = [CP_VERSION.strip(), REFERENCE_CONDITION.strip()]
    header_lines += header
    header = '\n'.join(header_lines[:2]) + '\n' + ' '.join(header_lines[2:])
    np.savetxt(output_path, data, delimiter=' ', header=header, fmt='%.6e', comments='', footer='/')

def write_oil_pvtx(pvt_array, solub_array, ref_oil_density, ref_gas_density, pvt_type, fluid, output_dir):
    filename = f"{pvt_type}_{fluid}_CP.INC"
    output_path = os.path.join(output_dir, filename).upper()
    pressure = pvt_array[:, 1] / BARS_TO_PASCALS
    fluid_density = pvt_array[:, 2]

    fluid_viscosity = pvt_array[:, 3] * 1e3
    fluid_fvf = ref_oil_density / fluid_density

    x_oil = solub_array[:, 2]
    x_gas = solub_array[:, 3]

    rs = (x_gas * GAS_MW * ref_oil_density) / (WATER_MW * (ref_gas_density - (x_gas * GAS_MW))) #mass fraction

    if pvt_type == 'PVDO':
        header = ['-- Pressure [bar]', 'Fluid FVF [rm³/sm³]', 'Viscosity [cP]\nPVDO']
        data = np.column_stack((pressure, fluid_fvf, fluid_viscosity))
        print(f' Msg: Dead oil PVT is generated')
    elif pvt_type == 'PVTW':
        header = ['-- Pressure [bar]', 'Fluid FVF [rm³/sm³]', 'Viscosity [cP]\nPVTW']
        data = np.column_stack((pressure, fluid_fvf, fluid_viscosity))
        print(f' Msg: Water PVT is generated')
    elif pvt_type == 'PVTO':
        fluid_fvf = ref_oil_density / fluid_density / (1 - x_oil)
        data = np.column_stack((rs, pressure, fluid_fvf, fluid_viscosity))
        with (open(output_path, 'w') as f):
            f.write(CP_VERSION)
            f.write(REFERENCE_CONDITION)
            f.write('-- RS [m³/m³] Pressure [bar] Fluid FVF [rm³/sm³] Viscosity [cP]\nPVTO\n')
            for i in range(data.shape[0]):
                f.write(f"{rs[i]:.6e}\t{pressure[i]:.6e}\t{fluid_fvf[i]:.6e}\t{fluid_viscosity[i]:.6e}\n")
                # todo: after double-checking, include the remaining part for the unsaturated zone

        print(f' Msg: Live oil PVT with vaporized gas is generated')
        return
    elif pvt_type == 'PVCO':
        header = ['-- Pressure [bar]', 'RS [m³/m³]', 'Fluid FVF [rm³/sm³]', 'Fluid compressibility [1/bar]\nPVCO']
        fluid_compressibility = 3e-5
        data = np.column_stack((pressure, rs, fluid_fvf, fluid_viscosity, fluid_compressibility))
        print(f' Msg: Live oil fluid PVT is generated')
    header_lines = [CP_VERSION.strip(), REFERENCE_CONDITION.strip()]
    header_lines += header
    header = '\n'.join(header_lines[:2]) + '\n' + ' '.join(header_lines[2:])
    np.savetxt(output_path, data, delimiter=' ', header=header, fmt='%.6e', comments='', footer='/')


def main():
    parser = argparse.ArgumentParser(description='This script generates fluid flow models for each problem.')
    parser.add_argument('-c', '--case', default='.',
                        help='Identification of the SPE11 benchmark case: spe11a, spe11b or spe11c.')

    parser.add_argument('-f', '--folder', default='.',
                        help='Path to the folder containing the simulation input files.')

    args = parser.parse_args()
    case_name = args.case
    case_config = CASE_CONFIG[case_name]
    output_dir = args.folder
    print(' \n===========================================================================================')
    print(CP_VERSION)
    print(REFERENCE_CONDITION)
    print(f" Msg: Fluid Name={REFERENCE_GAS}, Density={CP.PropsSI('D', 'P', STD_PRESSURE, 'T', STD_TEMPERATURE, REFERENCE_GAS)} kg/m³")
    print(f" Msg: Fluid Name={REFERENCE_WATER}, Density={CP.PropsSI('D', 'P', STD_PRESSURE, 'T', STD_TEMPERATURE, REFERENCE_WATER)} kg/m³")

    # SPE11 specific cases
    pvt_data_points = case_config['pvt_data_points']
    temperature = case_config['temperature']
    min_pressure = case_config['min_pressure']
    max_pressure = case_config['max_pressure']

    report_folder = os.path.join(output_dir, case_name)
    os.makedirs(report_folder, exist_ok=True)

    # Generate RELA and scale saturation table
    all_rela = generate_fluid_rela(case_name)

    header = ['-- SG [-] KRG [-] KROG [-] PCOG [bar]\nSGOF']
    # write all rela into a single file
    write_data(all_rela, header, report_folder, REFERENCE_GAS, 'RELA', ' ', case_name)

    co2_pvt_data = get_fluid_properties(temperature, min_pressure, max_pressure, pvt_data_points, REFERENCE_GAS)
    h2o_pvt_data = get_fluid_properties(temperature, min_pressure, max_pressure, pvt_data_points, REFERENCE_WATER)
    co2_solubility_data = get_fluid_solubility(co2_pvt_data)

    header = ['temperature [K]', 'pressure [Pa]', 'density [kg/m³]', 'viscosity [Pa.s]', 'enthalpy [J/kg]',
              'thermal_conductivity [W/m.K]', 'cv [J/kg.K]', 'cp [J/kg.K]']

    write_data(co2_pvt_data, header, report_folder, REFERENCE_GAS, 'PVTx', ',', case_name)
    write_data(h2o_pvt_data, header, report_folder, REFERENCE_WATER, 'PVTx', ',', case_name)

    header = ['temperature [K], pressure [Pa], y_H2O [mol/mol], x_CO2 [mol/mol]']
    write_data(co2_solubility_data, header, report_folder, REFERENCE_GAS, 'solubility', ',', case_name)
    write_gas_pvtx(co2_pvt_data, co2_solubility_data, REFERENCE_GAS_DENSITY, REFERENCE_WATER_DENSITY, 'PVTG','CO2', report_folder)
    write_oil_pvtx(h2o_pvt_data, co2_solubility_data, REFERENCE_WATER_DENSITY, REFERENCE_GAS_DENSITY, 'PVDO','H2O', report_folder)
    print(' ===========================================================================================\n')

if __name__ == '__main__':
    main()