"""

__author__ = "fgasa"

"""
import argparse
import numpy as np
import os

CASE_CONFIG = {
    'spe11a': {
        'zdatum': 0,
        'dx': [0.01] * 280,
        'dy': [0.01] * 1,
        'dz': [0.01] * 120
    },
    'spe11b': {
        'zdatum': 0,
        'ni': 842,
        'nj': 1,
        'nk': 120,
    },
    'spe11c': {
        'zdatum': 0,
        #Gaussian function to approximate the shape of an anticline
        'sigmax': 1000000,
        'sigmay': 3000,
        'amplitude': 500
    }
}


class MeshGenerator:
    def __init__(self, output_dir, xcoord, ycoord, zcoord,XCOORD, YCOORD, ZCOORD):
        '''Initialize the MeshGenerator with starting coordinates and grid setup'''

        assert XCOORD.shape == YCOORD.shape == ZCOORD.shape, 'XCOORD, YCOORD, and ZCOORD must have the same shape'

        self.output_dir = output_dir
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.zcoord = zcoord

        self.XCOORD = XCOORD
        self.YCOORD = YCOORD
        self.ZCOORD = ZCOORD

        # Mesh dimensions
        self.NI, self.NJ, self.NK = len(self.xcoord), len(self.ycoord), len(self.zcoord)
        self.ni, self.nj, self.nk = self.NI - 1, self.NJ - 1, self.NK - 1
        self.actnum = self.ni * self.nj * self.nk
        print(f' Msg: Grid dimension (nI,nJ,nK): {self.ni}, {self.nj}, {self.nk}')
        print(f' Msg: Total elements in model: {self.actnum}')

    def generate_coord(self):
        # Creating the COORD array for file output
        X1 = np.tile(self.xcoord, self.NJ).reshape(-1, 1)
        Y1 = np.repeat(self.ycoord, self.NI).reshape(-1, 1)

        Z1 = self.ZCOORD[:, :, 0].reshape(self.NI * self.NJ, 1)
        Z2 = self.ZCOORD[:, :, -1].reshape(self.NI * self.NJ, 1)

        self.COORD = np.hstack((X1, Y1, Z1, X1, Y1, Z2))
        print(f' Msg: Mesh coordinates processed successfully')

    def generate_zcorn(self):
        '''Generate the ZCORN array based on mesh coordinates'''
        self.ZCORN = []
        for k in range(self.nk): #Loop through layers
            for j in range(self.nj): # Loop through rows for the current layer
                ZCOORD_k_T = self.ZCOORD[:, :, k]
                ZCOORD_k_T_j = ZCOORD_k_T[j, :]
                ZCOORD_k_T_j_1 = ZCOORD_k_T[j + 1, :]

                ZCOORD_k_T_j_NW = ZCOORD_k_T_j[:-1]
                ZCOORD_k_T_j_NE = ZCOORD_k_T_j[1:]
                ZCOORD_k_T_j_SW = ZCOORD_k_T_j_1[:-1]
                ZCOORD_k_T_j_SE = ZCOORD_k_T_j_1[1:]

                ZCORN_k_T_j = np.kron(ZCOORD_k_T_j_NW, [1, 0]) + np.kron(ZCOORD_k_T_j_NE, [0, 1])
                ZCORN_k_T_j_1 = np.kron(ZCOORD_k_T_j_SW, [1, 0]) + np.kron(ZCOORD_k_T_j_SE, [0, 1])
                self.ZCORN.extend([ZCORN_k_T_j, ZCORN_k_T_j_1])

            for j in range(self.nj):
                ZCOORD_k_B = self.ZCOORD[:, :, k + 1]
                ZCOORD_k_B_j = ZCOORD_k_B[j, :]
                ZCOORD_k_B_j_1 = ZCOORD_k_B[j + 1, :]

                ZCOORD_k_B_j_NW = ZCOORD_k_B_j[:-1]
                ZCOORD_k_B_j_NE = ZCOORD_k_B_j[1:]
                ZCOORD_k_B_j_SW = ZCOORD_k_B_j_1[:-1]
                ZCOORD_k_B_j_SE = ZCOORD_k_B_j_1[1:]

                ZCORN_k_B_j = np.kron(ZCOORD_k_B_j_NW, [1, 0]) + np.kron(ZCOORD_k_B_j_NE, [0, 1])
                ZCORN_k_B_j_1 = np.kron(ZCOORD_k_B_j_SW, [1, 0]) + np.kron(ZCOORD_k_B_j_SE, [0, 1])
                self.ZCORN.extend([ZCORN_k_B_j, ZCORN_k_B_j_1])
        print(f' Msg: ZCORN values generated successfully')

    def generate_zcorn_new(self):
        '''Generate the ZCORN array based on mesh coordinates'''
        self.ZCORN = []
        for k in range(self.nk): #Loop through layers
            for j in range(self.nj):
                self.ZCORN.extend(self._calculate_zcorn(k, j))  # Loop through rows for the current layer
                self.ZCORN.extend(self._calculate_zcorn(k + 1, j))  # Next layer

        print(' Msg: ZCORN values generated successfully')

    def _calculate_zcorn(self, k, j):
        '''Helper function to calculate ZCORN for a specific layer'''

        # Extract Z coordinates for layer k at row j and j+1
        Z_k_j, Z_k_j1 = self.ZCOORD[j, :, k], self.ZCOORD[j + 1, :, k]

        # Extract Northwest, Northeast, Southwest, Southeast corners
        NW, NE = Z_k_j[:-1], Z_k_j[1:]
        SW, SE = Z_k_j1[:-1], Z_k_j1[1:]

        # Interpolate ZCORN values (top and bottom faces)
        Z_top = np.kron(NW, [1, 0]) + np.kron(NE, [0, 1])
        Z_bottom = np.kron(SW, [1, 0]) + np.kron(SE, [0, 1])

        return [Z_top, Z_bottom]

    def write_grdecl_file(self):

        mesh_filename = f'GRID_{self.ni}x{self.nj}x{self.nk}.GRDECL'
        self.meshfile = os.path.join(self.output_dir, mesh_filename)

        with open(self.meshfile, 'w') as f:
            license_text = '''-- This reservoir simulation deck is made available under the Open Database\n-- License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in\n-- individual contents of the database are licensed under the Database Contents\n-- This reservoir simulation deck is made available under the Open Database\n-- License: http://opendatacommons.org/licenses/dbcl/1.0/ \n'''
            f.write(license_text)
            f.write('\n\n-- Copyright (C) 2024 CAU (Geohydromodelling at Kiel University)\n\n\n\n')
            # --PINCHXY
            f.write('PINCH\n /\n\nMAPUNITS\n  METRES /\n\nGRIDUNIT\n  METRES MAP /\n\n')
            f.write(f'SPECGRID\n  {self.ni} {self.nj} {self.nk} 1 F /\n\n')
            # f.write(f'COORDSYS\n   1 {self.nz} /\n\n')

            f.write('COORD\n')
            np.savetxt(f, self.COORD, fmt='%.2f', delimiter=' ')
            f.write('/\n\n')

            f.write('ZCORN\n')

            np.savetxt(f, np.array(self.ZCORN).reshape(-1, 10), fmt='%.2f', delimiter=' ') #fmt='%.6e',
            f.write('/\n\nACTNUM\n')
            f.write(f'  {self.actnum}*1 /\n')

        print(' Msg: Grid file is writted in ', self.meshfile)

    def write_cmg_grid_file(self):

        mesh_filename = f'GRID_{self.ni}x{self.nj}x{self.nk}_CMG.dat'
        self.meshfile = os.path.join(self.output_dir, mesh_filename)

        with open(self.meshfile, 'w') as f:
            license_text = '''** This reservoir simulation deck is made available under the Open Database\n** License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in\n** individual contents of the database are licensed under the Database Contents\n** This reservoir simulation deck is made available under the Open Database\n** License: http://opendatacommons.org/licenses/dbcl/1.0/ \n'''
            f.write(license_text)
            f.write('**\n')
            f.write('**\n')
            f.write('** Copyright (C) 2024 CAU (Geohydromodelling at Kiel University)\n')
            f.write('**\n')
            f.write('**\n')
            f.write(f'GRID CORNER {self.ni} {self.nj} {self.nk}\n')

            f.write('COORD\n')
            # todo
            CMG_COORD = np.zeros((self.COORD.shape[0], 6))

            # Fill the new array
            CMG_COORD[:, 0] = self.COORD[:, 0]
            CMG_COORD[:, 1] = self.COORD[:, 1]
            CMG_COORD[:, 2] = self.COORD[:, 2]
            CMG_COORD[:, 3] = self.COORD[:, 3]
            CMG_COORD[:, 4] = self.COORD[:, 4]
            CMG_COORD[:, 5] = self.COORD[:, 5]

            np.savetxt(f, CMG_COORD, fmt='%.6e',  delimiter=' ')
        print(' Msg: Grid file is writted in ', self.meshfile)


    def generate_mesh(self):
        self.generate_coord()
        self.generate_zcorn()
        self.write_grdecl_file()
        self.write_cmg_grid_file()

def main():
    parser = argparse.ArgumentParser(description='This script generates Cartesian grid type using GRDECL format')
    parser.add_argument('-c', '--case', default='.', help='Identification of the SPE11 benchmark case: spe11a, spe11b or spe11c.')
    parser.add_argument('-f', '--folder', default='.', help='Path to directory for model geometry generating')
    parser.add_argument('-ni', '--ni_elements', type=int, default=280, help='Number of elements in the X direction')
    parser.add_argument('-nj', '--nj_elements', type=int, default=1, help='Number of elements in the Y direction')
    parser.add_argument('-nk', '--nk_elements', type=int, default=120, help='Number of elements in the Z direction')
    parser.add_argument('-dx', '--dx', type=float, default=1, help='Element size in the X direction')
    parser.add_argument('-dy', '--dy', type=float, default=1, help='Element size in the Y direction')
    parser.add_argument('-dz', '--dz', type=float, default=1, help='Element size in the Z direction')
    parser.add_argument('-z0', '--z_datum', type=float, default=0, help='Reference depth')

    args = parser.parse_args()
    work_dir = args.folder
    case_name = args.case
    assert case_name in CASE_CONFIG, f' Error: Invalid case name: {case_name}'
    case_config = CASE_CONFIG[case_name]

    nI = args.ni_elements
    nJ = args.nj_elements
    nK = args.nk_elements
    dx = args.dx
    dy = args.dy
    dz = args.dz
    z_datum = args.z_datum

    output_dir = os.path.join(work_dir, str(case_name))
    os.makedirs(output_dir, exist_ok=True)
    print(' \n===========================================================================================')
    print(f' Msg: Work dir: {work_dir}')
    print(f' Msg: Output dir: {output_dir}')
    print(f' Msg: Case name: {case_name}')

    if case_name == 'spe11b' or case_name == 'spe11c':
        dx_all = [1] * 1 + [dx-1] + [dx] * (nI - 4) + [dx-1] * 1 + [1] *1
    else:
        dx_all = [dx] * nI
    dy_all = [dy] * nJ
    dz_all = [dz] * nK

    usr_xcoord = np.concatenate([[0], 0 + np.cumsum(dx_all)])
    usr_ycoord = np.concatenate([[0], 0 + np.cumsum(dy_all)])
    usr_zcoord = np.concatenate([[z_datum], z_datum + np.cumsum(dz_all)])

    XCOORD, YCOORD, ZCOORD = np.meshgrid(usr_xcoord, usr_ycoord, usr_zcoord)
    if case_name == 'spe11c':
        X, Y = np.meshgrid(usr_xcoord, usr_ycoord)
        # todo, fix equation
        Z0 = z_datum - case_config['amplitude'] * np.exp(-((X - (usr_xcoord[-1] - usr_xcoord[0]) / 2) ** 2 / (2 * case_config['sigmax'] ** 2) +
                                   (Y - (usr_ycoord[-1] - usr_ycoord[0]) / 2) ** 2 / (2 * case_config['sigmay'] ** 2)))

        surface_smoother = 0.1
        Z0[np.abs(Z0 - z_datum) < surface_smoother] = z_datum

        ZCOORD[:, :, 0] = Z0

        for i in range(nK):
            ZCOORD[:, :, i + 1] = ZCOORD[:, :, i] + dz_all[i]
    else:
       ZCOORD[:, :, 0] = z_datum

    mesh_generator = MeshGenerator(output_dir, usr_xcoord, usr_ycoord, usr_zcoord, XCOORD, YCOORD, ZCOORD )
    mesh_generator.generate_mesh()
    print(' ===========================================================================================\n')

if __name__ == '__main__':
    main()