import numpy as np
import scipy.linalg as sl
import pickle
import warnings
import argparse
import sys
import pickle
import time
import os
from pyscf import gto, scf
from scipy.linalg import block_diag

from ao2mo import ao2mo
from makeH import BuildHamitonian
from Spin_Proj import Spin_Proj
from PG_Proj import PG_Proj
from PCC_utils import FrozenCore
from MatchOrb import MatchOrb

# from Cmplx_Proj import *

"""
This code establishes many quantities needed for PHF and PCC. It begins by
gathering molecular information and getting integrals. It will also run a RHF
calculations as the orbitals for later HF and PHF calculations will be written
in the basis of the RHF orbitals, referred to here as the OAO(orthogonal atomic
oribtal) orbitals.

The OAO orbitals can be read in by giving a read_OAO file. During a scan, the
RHF orbitals at the current geometry will be calculated, the rotated to match
the given OAO as best as possible.

This code also establishes the projection operator for spin and point group
projection.

The last thing this code does is a frozen core projection. This is useful since
this code stores all integrals in full, which is the major limitation of this
code.
"""

elements = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
}


def xyz_reader(file, scan, VERBOSE=1):
    """Read xyz file for geometry"""

    try:
        with open(file, "r") as f:
            lines = f.readlines()

    except FileNotFoundError:
        raise FileNotFoundError(
            "xyz file does not exist! Cannot proceed without geometry!"
        )

    try:
        n = int(lines[0].split()[0])
    except ValueError:
        raise TypeError("First line of xyz is not an integer!")
    if n < 1:
        raise ValueError("Cannot have fewer than one atom!")

    if scan == 0:
        if len(lines) < n + 2:
            raise ValueError(
                "xyz file does not contain enough lines for the number of atoms!"
            )
        if (len(lines) > n + 2) and (VERBOSE > 0):
            warnings.warn(
                "xyz file contains more lines than indicated by atom number, make sure your xyz file is correct"
            )
    else:
        if len(lines) < 2 * n + 4:
            raise ValueError(
                "xyz file does not contain enough lines for the number of atoms and scan coord!"
            )
        if (len(lines) > 2 * n + 4) and (VERBOSE > 0):
            warnings.warn(
                "xyz file contains more lines than indicated by atom number, make sure your xyz file is correct"
            )

    atom_num = np.empty(n, dtype=int)
    coord = np.empty((n, 3), dtype=float)

    for i in range(n):
        cur = lines[i + 2]
        cur = cur.split()
        if len(cur) < 4:
            raise ValueError(f"recieved insufficient number of items for atom {i+1}")
        elif len(cur) > 4:
            warnings.warn(
                f"atom {i+1} in xyz file contains more than four items, make sure your xyz file is correct"
            )

        try:
            atom = elements[cur[0].upper()]
        except KeyError:
            raise ValueError(f"atom {i+1} is not in the elements dictionary!")

        atom_num[i] = atom

        try:
            coord[i, :] = np.array([float(x) for x in cur[1:4]])
        except ValueError:
            raise ValueError(
                f"Recieved value that cannot be interpreted as float in coordinates of atom {i+1}"
            )
    if scan == 0:
        return atom_num, coord, None

    displace = np.empty((n, 3), dtype=float)

    for i in range(n):
        cur = lines[i + 4 + n]
        cur = cur.split()
        if len(cur) < 3:
            raise ValueError(
                f"recieved insufficient number of items for atom {i+1} in scan"
            )
        elif len(cur) > 3:
            warnings.warn(
                f"atom {i+1} in xyz file contains more than four items in scan, make sure your xyz file is correct"
            )

        try:
            displace[i, :] = np.array([float(x) for x in cur[:3]])
        except ValueError:
            raise ValueError(
                f"Recieved value that cannot be interpreted as float in coordinates of atom {i+1}"
            )

    return atom_num, coord, displace


def prepare_PCC(settings, Values, step=0, read_OAO=None, write_OAO=None):
    "Prepare molecule information"
    if settings.HamType == "Mol":
        Values.mol = gto.Mole()
        atom_num, coord, displace = xyz_reader(settings.geom_file, settings.scan)
        if settings.scan > 0:
            coord += step*displace
        Values.mol.atom = []
        for i in range(len(atom_num)):
            Values.mol.atom.append([atom_num[i], coord[i, :]])
        Values.mol.spin = 0
        Values.mol.basis = settings.basis[0]
        Values.mol.unit = settings.unit
        Values.mol.build()

        ovlp = Values.mol.intor("int1e_ovlp")
        Values.h1 = Values.mol.intor("int1e_kin") + Values.mol.intor("int1e_nuc")
        Values.eri = Values.mol.intor("int2e")
        Values.NAO = Values.mol.nao
        Values.NOccSO = sum(Values.mol.nelec)
        Values.Enuc = Values.mol.energy_nuc()
    elif settings.HamType == "Hub":
        mol = gto.Mole()
        mol.nelectron = settings.nele
        nsite = settings.nx * settings.ny
        Values.h1, Values.eri = BuildHamitonian(
            settings.nx, settings.ny, settings.U0, pbcx=True, pbcy=True
        )
        ovlp = np.eye(nsite)
        Values.Enuc = 0
        Values.NAO = nsite
        Values.NOccSO = nele

    # Do RHF
    "Calculate S^-1/2 for canonical symmetrization"
    tol = 1e-10
    evals, evecs = sl.eigh(ovlp, lower=False)
    invS = [np.sqrt(abs(x)) / (x + tol) for x in evals]
    Sinv = np.diag(invS)
    OrthAO = evecs.dot(Sinv)
    Xinv = np.linalg.inv(OrthAO)

    "Define variables for GHF form"
    # TODO Remove redundant variables
    Values.NSO = 2 * Values.NAO
    Values.NVrtSO = Values.NSO - Values.NOccSO
    Values.NOccAO = Values.NOccSO // 2
    Values.OV = Values.NOccSO * Values.NVrtSO
    Values.NOccA = Values.NOccAO
    Values.NVrtA = Values.NAO - Values.NOccA

    """ 
    Spin Projection, SP = 0, No projection
                     SP = 1, SGHF
                     SP = 2, SUHF
                     SP = 3, SzPHF
    for SUHF & SzPHF, set grid pts as [1,n]
    """
    (
        Values.ncisp,
        Values.roota,
        Values.rootb,
        Values.rooty,
        Values.weightsp,
        Values.R1,
        Values.R2,
    ) = Spin_Proj(settings.SP, settings.ngrid, Values.NAO, Values.NSO, settings.J)

    """
    Point Group Projection

    """
    Values.ncipg, Values.weightpg, Values.Rpg = PG_Proj(
        Values.mol,
        settings.PG,
        settings.Irrep,
        ovlp,
        OrthAO,
        Values.NAO,
        nx=settings.nx,
        ny=settings.ny,
        kx=settings.kx,
        ky=settings.ky,
    )
    Values.npg = len(Values.weightpg)

    """
    Complex Conj = 0, real
                 = 1, complex
                 = 2, complex projection
    """
    if settings.CmplxConj == 2:
        Values.ncik = 2
    else:
        Values.ncik = 1

    Values.nci = Values.ncisp * Values.ncipg * Values.ncik
    Values.npoints = settings.ngrid[0] * settings.ngrid[1] * Values.npg

    if Values.rank == 0:
        ## RHF(Orbitals in the future are written in the basis of these RHF orbitals)
        RHF = scf.RHF(Values.mol)
        RHF.diis_space = 20
        RHF.verbose = 0
        if read_OAO:
            if settings.VERBOSE > 1:
                print(f"Loading OAO from {read_OAO}")
            Values.OAO = pickle.load(open(f"{read_OAO}", "rb"))
        else:
            if step == 0:
                RHF.kernel()
                Values.OAO = RHF.mo_coeff
            else:
                "Update OAO to new geometry"
                # den = Values.OAO[:,:Values.NOccAO].dot(Values.OAO[:,:Values.NOccAO].T)
                RHF.kernel()
                Values.OAO = MatchOrb(RHF.mo_coeff, Values.OAO, Values.NOccAO)
                # _,Values.OAO,_ = pickle.load(open(f"temp1", "rb"))

            if write_OAO:
                pickle.dump(Values.OAO, open(f"{write_OAO}", "wb"))

    if settings.use_MPI == True:
        if Values.rank != 0:
            Values.OAO = np.empty((Values.NAO, Values.NAO),dtype=float, order="F")
        Values.comm.Bcast(Values.OAO, root=0)


    # update h1 for x2c
    # Values.h1 = RHF.get_hcore()

    if ((settings.NFC > 0) or (settings.NFV > 0)):
        "Remove frozen orbitals as defined by NFC and NFV"
        E0, Values.h1, Values.eri = FrozenCore(
            Values.h1, Values.eri, Values.OAO, settings.NFC, settings.NFV, Values.NAO
        )
    else:
        Values.h1 = ao2mo(Values.h1, Values.OAO, 2)
        Values.eri = ao2mo(Values.eri, Values.OAO, 4)
        E0 = 0.

    "redefine parameters"
    NAOnew = Values.NAO - settings.NFC - settings.NFV
    NOccSOnew = Values.NOccSO - 2 * settings.NFC
    Values.Enuc += E0
    if settings.VERBOSE > 0:
        print("FC ENuc=", Values.Enuc)

    "transfer PG symm operators"
    if settings.HamType == "Mol":
        ovlp = Values.mol.intor("int1e_ovlp")
    elif settings.HamType == "Hub":
        ovlp = np.eye(settings.nx * settings.ny)
    Values.ncipg, Values.weightpg, Values.Rpg = PG_Proj(
        Values.mol,
        settings.PG,
        settings.Irrep,
        ovlp,
        Values.OAO,
        Values.NAO,
        nx=settings.nx,
        ny=settings.ny,
        kx=settings.kx,
        ky=settings.ky,
    )
    Rpgnew = np.zeros([Values.npg, 2 * NAOnew, 2 * NAOnew], dtype=complex)
    for i in range(Values.npg):
        tmp = Values.Rpg[
            i,
            settings.NFC : Values.NAO - settings.NFV,
            settings.NFC : Values.NAO - settings.NFV,
        ]
        Rpgnew[i, :, :] = block_diag(tmp, tmp)
    Values.Rpg = Rpgnew.copy()
    Values.NAO = NAOnew
    Values.NOccSO = NOccSOnew
    Values.NSO = 2 * Values.NAO
    Values.NVrtSO = Values.NSO - Values.NOccSO
    Values.NOccAO = Values.NOccSO // 2
    Values.OV = Values.NOccSO * Values.NVrtSO
    Values.NOccA = Values.NOccAO
    Values.NVrtA = Values.NAO - Values.NOccA
    Values.mol.nelectron = Values.NOccSO
    ovlp = np.eye(Values.NAO)

    OrthAO = np.eye(Values.NAO)
    Xinv = np.linalg.inv(OrthAO)
    Values.Ovlp = ao2mo(ovlp, OrthAO, 2)
    Values.h1 = ao2mo(Values.h1, OrthAO, 2)
    Values.eri = ao2mo(Values.eri, OrthAO, 4)

    (
        Values.ncisp,
        Values.roota,
        Values.rootb,
        Values.rooty,
        Values.weightsp,
        Values.R1,
        Values.R2,
    ) = Spin_Proj(settings.SP, settings.ngrid, Values.NAO, Values.NSO, settings.J)

    return settings, Values
