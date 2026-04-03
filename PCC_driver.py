#!/usr/bin/python

"""
This code performs the PHF and PCC optimization for molecules and Hubbard
lattices, run through the driver function below. This code contains
contributions from Carlos Jimenez-Hoyos, Yiheng Qiu, Tom Henderson, and
Ruiheng Song.

To use this code, prepare a json file containing settings for the calculation,
see comments in PCC_objects settings class for more details. Some settings can be
set by command line argument, see the driver function below for more details.
The geometry for the calculation is to be stored in a xyz file specified in the
settings.

Print level is controlled with VERBOSE option in settings, default is 1.
"""

import numpy as np
import scipy.special as ss
import argparse
import sys
import os


import PCC_objects
import PCC_setup
import PCC_orbs
import PCC_run_PHF
import PCC_run_PCC

np.set_printoptions(precision=14, threshold=np.inf, suppress=True, linewidth=200000000)


def parse(args):
    "Set up argparse that does not conflict with unittests"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--settings",
        type=str,
        help="json file containing settings for calculation",
    )
    parser.add_argument(
        "-g", "--geometry", type=str, help="xyz file containing molecular geometry"
    )
    parser.add_argument("-b", "--basis", type=str, help="basis set for calculation")
    args = parser.parse_args()
    return args


def driver(file=None):
    """
    Command Line Arguments
    ---------------------------------------
    settings/s : str
        json file containing settings
    geometry/g : str
        xyz file containing geometry for calculation
    basis/b : str
        basis to be used for calculation(must be in pyscf)
    """

    "Establish settings"
    if file == None:
        args = parse(sys.argv[1:])
        if args.settings:
            file = args.settings
    else:
        args = None

    print("Reading and Checking Settings")
    settings = PCC_objects.settings(file, args)
    if settings.VERBOSE > 1:
        print("Settings are Established and Valid")

    Values = PCC_objects.Values()

    if settings.use_MPI:
        from mpi4py import MPI

        Values.comm = MPI.COMM_WORLD
        Values.fcomm = MPI.COMM_WORLD.py2f()
        Values.rank = Values.comm.Get_rank()
    else:
        Values.comm = None
        Values.fcomm = None
        Values.rank = 0

    "Only main process should print to output"
    if Values.rank != 0:
        settings.VERBOSE = 0

    if settings.scan == 0:
        """
        Single point Calculation
        """
        if (settings.read_PHF == True) and (
            os.path.exists(f"{settings.mol_name}_SGHF.p")
        ):
            "Remove setup basis sets"
            settings.basis = [settings.basis[-1]]
            read_OAO = f"{settings.mol_name}_OAO.p"
            read_PHF = f"{settings.mol_name}_SGHF.p"
            convert = False
        else:
            read_OAO = None
            read_PHF = None
            if len(settings.basis) > 1:
                convert = True
            else:
                convert = False

        if (settings.read_PCC == True) and (
            os.path.exists(f"{settings.mol_name}_PCC.p")
        ):
            read_PCC = f"{settings.mol_name}_PCC.p"
        else:
            read_PCC = None

        if settings.chkpoint:
            write_OAO = f"{settings.mol_name}_OAO.p"
            write_PHF = f"{settings.mol_name}_SGHF.p"
            write_PCC = f"{settings.mol_name}_PCC.p"
        else:
            write_OAO = None
            write_PHF = None
            write_PCC = None

        "Set up should only be done by main computer"
        settings, Values = PCC_setup.prepare_PCC(
            settings, Values, read_OAO=read_OAO, write_OAO=write_OAO
        )

        if False:
            # FCI
            from pyscf import fci
            mci = fci.direct_spin1.FCI()
            mci = fci.addons.fix_spin_(mci, ss=settings.J*(settings.J+1))
            fcie, fcivec = mci.kernel(Values.h1, Values.eri, Values.NAO, Values.NOccSO, nroots=4)
            print("E(FCI1)= %8.8f" %(fcie[0]+Values.Enuc))
            print("E(FCI2)= %8.8f" %(fcie[1]+Values.Enuc))
            raise

        if not ((settings.read_PHF == True) and (read_OAO != None)):
            if Values.rank == 0:
                Values = PCC_orbs.prepare_orbs(settings, Values)
                temp = Values.A2G.dtype.str
            else:
                temp = ""
            if settings.use_MPI == True:
                temp = Values.comm.bcast(temp, root=0)
                if Values.rank != 0:
                    Values.A2G = np.empty(
                        (Values.NSO, Values.NSO), dtype=temp, order="C"
                    )
                Values.comm.Bcast(Values.A2G, root=0)

        if False:
            # UCCSD
            from pyscf import cc, scf
            from PCC_utils import SortOrb
            HF = scf.UHF(Values.mol)
            m,n = Values.A2G.shape
            A = SortOrb(Values.A2G, Values.NAO, Values.NOccAO, Values.NSO, Values.NOccSO, 0)
            MO = [A[:m//2,:n//2],A[m//2:,n//2:]]
            dm1 = np.dot(MO[0], MO[0].T)
            dm2 = np.dot(MO[1], MO[1].T)
            dm = np.array([dm1, dm2])
            HF.kernel(dm)
            mfcc = cc.CCSD(HF)
            mfcc.direct= True
            mfcc.run()
            raise

        Values = PCC_run_PHF.run_PHF(
            settings,
            Values,
            write_OAO=write_OAO,
            read_PHF=read_PHF,
            write_PHF=write_PHF,
        )
        if settings.do_CC:
            Values = PCC_run_PCC.run_PCC(
                settings,
                Values,
                convert=convert,
                read_PCC=read_PCC,
                write_PCC=write_PCC,
            )
    else:
        """
        Geometry Scan
        """
        for n in range(settings.scan + 1):
            if n == 1:
                "Remove setup basis sets"
                settings.basis = [settings.basis[-1]]

            if (settings.read_PHF == True) and (
                os.path.exists(f"{settings.mol_name}_{n:02d}_SGHF.p")
            ):
                read_OAO = f"{settings.mol_name}_{n:02d}_OAO.p"
                read_PHF = f"{settings.mol_name}_{n:02d}_SGHF.p"
                settings.basis = [settings.basis[-1]]
                convert = False
            else:
                read_OAO = None
                read_PHF = None
                if len(settings.basis) > 1:
                    convert = True
                else:
                    convert = False
            if n > 0:
                convert = False

            if (settings.read_PCC == True) and (
                os.path.exists(f"{settings.mol_name}_{n:02d}_PCC.p")
            ):
                read_PCC = f"{settings.mol_name}_{n:02d}_PCC.p"
            else:
                read_PCC = None

            if settings.chkpoint:
                write_OAO = f"{settings.mol_name}_{n:02d}_OAO.p"
                write_PHF = f"{settings.mol_name}_{n:02d}_SGHF.p"
                write_PCC = f"{settings.mol_name}_{n:02d}_PCC.p"
            else:
                write_OAO = None
                write_PHF = None
                write_PCC = None

            settings, Values = PCC_setup.prepare_PCC(
                settings, Values, step=n, read_OAO=read_OAO, write_OAO=write_OAO
            )

            if (n == 0) and (not ((settings.read_PHF == True) and (read_OAO != None))):
                if Values.rank == 0:
                    Values = PCC_orbs.prepare_orbs(settings, Values)
                    temp = Values.A2G.dtype.str
                else:
                    temp = ""
                if settings.use_MPI:
                    if settings.use_MPI == True:
                        temp = Values.comm.bcast(temp, root=0)
                        if Values.rank != 0:
                            Values.A2G = np.empty(
                                (Values.NSO, Values.NSO), dtype=temp, order="C"
                            )
                        Values.comm.Bcast(Values.A2G, root=0)

            Values = PCC_run_PHF.run_PHF(
                settings,
                Values,
                step=n,
                write_OAO=write_OAO,
                read_PHF=read_PHF,
                write_PHF=write_PHF,
            )
            if settings.do_CC:
                Values = PCC_run_PCC.run_PCC(
                    settings,
                    Values,
                    step=n,
                    convert=convert,
                    read_PCC=read_PCC,
                    write_PCC=write_PCC,
                )

    return Values


if __name__ == "__main__":
    driver()
