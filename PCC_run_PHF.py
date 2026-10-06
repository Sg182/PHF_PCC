import numpy as np
import scipy.linalg as sl
import pickle
import time
import os
import copy
from scipy.linalg import block_diag
from scipy.optimize import minimize, basinhopping
try:
    from ipopt import minimize_ipopt          # cyipopt <= 0.2 exposes it as `ipopt`
except ImportError:
    try:
        from cyipopt import minimize_ipopt    # cyipopt >= 1.0
    except ImportError:
        minimize_ipopt = None
import pyscf
import PCC_orbs

from ao2mo import ao2mo

# from Spin import *
# from LinDep import *
from makeH import SemiCanon, Mulliken2Dirac
from Spin_Proj import Spin_Proj
from PG_Proj import PG_Proj
from MatchOrb import MatchOrb
from PCC_run_PCC import run_PCC,convert_amp_basis
from ao2mo import ao2mo
from pyscf import fci

# from PHFTools import *
from PCC_utils import (
    FixGauge,
    BuildSx,
    BuildSy,
    BuildSz,
    EigenSolver,
    VecDecomp,
    FixCmplxGauge,
    LocalG,
    GetThoulessZ,
    Thouless2MOs,
    calcS,
    calcS2,
    FrozenCore,
    SortOrb,
)

"""
This code acts as the driver for the PHF code. Details of the algorithm can
be found in JCTC 14, 588 (2018).
"""


def convert_basis(settings, Values, new_basis, write_OAO=None):
    "This function converts the Values object to the basis specified"

    "Build new mol object"
    new_mol = pyscf.gto.Mole()
    new_mol.atom = Values.mol.atom
    new_mol.unit = Values.mol.unit
    new_mol.spin = Values.mol.spin
    new_mol.basis = new_basis
    new_mol.build()

    "Get overlap between basis sets and convert PHF orbitals"

    "Get RHF orbitals in new basis to act as orthogonal AO basis"
    RHF = pyscf.scf.RHF(new_mol)
    RHF.diis_space = 20
    RHF.verbose = 0
    RHF.kernel()

    "Get overlap between old and new basis then convert to RHF basis on each side"
    S_cross = pyscf.gto.mole.intor_cross("int1e_ovlp", new_mol, Values.mol)
    new_OAO = MatchOrb(RHF.mo_coeff, S_cross.dot(Values.OAO), Values.NOccSO)
    S_cross = new_OAO.T.dot(S_cross).dot(Values.OAO)

    "Up convert the SGHFMO coefficients"

    "Expand overlap to GHF form"
    n, o = S_cross.shape
    S = np.zeros([2 * n, 2 * o])
    S[:n, :o] = S_cross[:, :]
    S[n:, o:] = S_cross[:, :]

    "Convert occupied orbitals, renormalize, then build density to get virtuals"
    C = S.dot(Values.SGHFMO[:, : Values.NOccSO])
    ovp = C.T.conj().dot(C)
    val, vec = sl.eigh(ovp)
    s = 1.0 / np.sqrt(val)
    X = vec.dot(np.diag(s)).dot(vec.T.conj())
    C = C.dot(X)

    P = C.dot(C.T.conj())
    val, vec = sl.eigh(-P)
    if (settings.is_RHF) or (settings.SP == 2):
        "separate up and down orbitals"
        C_occ = vec[:, : Values.NOccSO]
        down = (
            np.diag(C_occ[: Values.NAO, :].T.conj().dot(C_occ[: Values.NAO, :])) == 0.0
        )
        up = np.invert(down)
        C_occ = np.hstack((C_occ[:, up], C_occ[:, down]))

        C_vir = vec[:, Values.NOccSO :]
        down = (
            np.diag(C_vir[: Values.NAO, :].T.conj().dot(C_vir[: Values.NAO, :])) == 0.0
        )
        up = np.invert(down)
        C_vir = np.hstack((C_vir[:, up], C_vir[:, down]))
    else:
        "Treat as GHF type orbitals"
        C_occ = vec[:, : Values.NOccSO]
        C_vir = vec[:, Values.NOccSO :]

    Values.SGHFMO = np.hstack((C_occ, C_vir))
    Values.OAO = new_OAO.copy()
    if settings.use_MPI == True:
        Values.comm.Bcast(Values.SGHFMO, root=0)
        Values.comm.Bcast(Values.OAO, root=0)

    "Update Values object"
    Values.mol = copy.deepcopy(new_mol)

    ovlp = Values.mol.intor("int1e_ovlp")
    Values.h1 = Values.mol.intor("int1e_kin") + Values.mol.intor("int1e_nuc")
    Values.eri = Values.mol.intor("int2e")
    Values.NAO = Values.mol.nao
    Values.NOccSO = sum(Values.mol.nelec)
    Values.Enuc = Values.mol.energy_nuc()
    Values.NSO = 2 * Values.NAO
    Values.NVrtSO = Values.NSO - Values.NOccSO
    Values.NOccAO = Values.NOccSO // 2
    Values.OV = Values.NOccSO * Values.NVrtSO
    Values.NOccA = Values.NOccAO
    Values.NVrtA = Values.NAO - Values.NOccA
    tol = 1e-10
    evals, evecs = sl.eigh(ovlp, lower=False)
    invS = [np.sqrt(abs(x)) / (x + tol) for x in evals]
    Sinv = np.diag(invS)
    OrthAO = evecs.dot(Sinv)
    Xinv = np.linalg.inv(OrthAO)

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
    # PG = "None"
    # Irrep = "Ag"
    # kx = 1
    # ky = 1
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

    Values.nci = Values.ncisp * Values.ncipg * Values.ncik
    Values.npoints = settings.ngrid[0] * settings.ngrid[1] * Values.npg

    if write_OAO and Values.rank == 0:
        pickle.dump(Values.OAO, open(f"{write_OAO}", "wb"))

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

    # "Update Values.A2G"
    # Values = PCC_orbs.prepare_orbs(settings, Values)
    # if settings.use_MPI == True:
    #     Values.comm.Bcast(Values.A2G, root=0)

    return Values


def get_ref(settings, Values, sci=False):
    """
    Get the real HF reference orbitals and thouless rotation that connect to
    Values.SGHFMO

    if sci is True, return Z in a form that can be fed into scipy optimizers
    """

    "Connect A2G to SGHFMO"
    Z, Olap = GetThoulessZ(Values.A2G, Values.SGHFMO, Values.NOccSO)

    if sci == True:
        Z = Z.flatten()
        z0 = np.zeros((2 * Values.NVrtSO * Values.NOccSO), dtype=float)
        z0[: Values.NVrtSO * Values.NOccSO] = Z.real
        z0[Values.NVrtSO * Values.NOccSO :] = Z.imag

        return z0, Olap
    else:
        return Z, Olap


def run_PHF(settings, Values, step=0, write_OAO=None, read_PHF=None, write_PHF=None):
    "Run the PHF calculation"

    # PHF
    if read_PHF:
        "If PHF is already done for PCC, update to newest basis and read old values from {settings.mol_name}_SGHF.p"
        if settings.VERBOSE > 1:
            print(f"Reading PHF values from {read_PHF}")
        Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk = pickle.load(
            open(f"{read_PHF}", "rb")
        )

        if False and Values.rank == 0:
            "Write orbitals for FCI PCC code"
            HOne = np.zeros([Values.NSO, Values.NSO])
            HTwo = np.zeros([Values.NSO, Values.NSO, Values.NSO, Values.NSO])
            HOne[: Values.NAO, : Values.NAO] = Values.h1[:, :]
            HOne[Values.NAO :, Values.NAO :] = Values.h1[:, :]
            HTwo[: Values.NAO, : Values.NAO, : Values.NAO, : Values.NAO] = Values.eri[
                :, :, :, :
            ]
            HTwo[Values.NAO :, Values.NAO :, Values.NAO :, Values.NAO :] = Values.eri[
                :, :, :, :
            ]
            HTwo[Values.NAO :, Values.NAO :, : Values.NAO, : Values.NAO] = Values.eri[
                :, :, :, :
            ]
            HTwo[: Values.NAO, : Values.NAO, Values.NAO :, Values.NAO :] = Values.eri[
                :, :, :, :
            ]
            HTwo = Mulliken2Dirac(HTwo)
            OrthAO = np.eye(Values.NAO)
            Xinv = np.linalg.inv(OrthAO)
            X = block_diag(OrthAO, OrthAO)
            Xinv = block_diag(Xinv, Xinv)
            C = SemiCanon(HOne, HTwo, Values.SGHFMO, Values.NOccSO, settings.SP)
            C = SortOrb(C,Values.NAO,Values.NOccAO,Values.NSO,Values.NOccSO,0)
            Ca = C[:Values.NAO,:Values.NAO]
            Cb = C[Values.NAO:,Values.NAO:]
            S = Ca.conj().T.dot(Cb)
            print (S)
            print(Values.Enuc)

            UHF = pyscf.scf.UHF(Values.mol)
            UHF.get_hcore = lambda *args: Values.h1
            UHF.get_ovlp = lambda *args: Values.Ovlp
            UHF._eri = pyscf.ao2mo.restore(8, Values.eri, Values.NAO)
            UHF.max_cycle = -1
            print(UHF.spin_square([Ca[:,:Values.NOccAO],Cb[:,:Values.NOccAO]],np.eye(Values.NAO)))
            dm1 = np.dot(
                Ca[:, : Values.NOccAO], Ca[:, : Values.NOccAO].T
            )
            dm2 = np.dot(
                Cb[:, : Values.NOccAO], Cb[:, : Values.NOccAO].T
            )
            dm = np.array([dm1, dm2])
            print(UHF.energy_tot(dm=dm) - Values.Enuc)
            # UHF.kernel(dm)

            h1a = np.einsum("pq,pi,qj->ij",Values.h1,Ca.conj(),Ca)
            h1b = np.einsum("pq,pi,qj->ij",Values.h1,Cb.conj(),Cb)
            h1_ = np.zeros((Values.NSO,Values.NSO),dtype=complex)
            h1_[::2,::2] = h1a
            h1_[1::2,1::2] = h1b

            h2 = Values.eri.transpose(0,2,1,3)
            h2 /= 2.0
            h2aa = np.einsum("pqrs,pi,qj,rk,sl->ijkl",h2,Ca.conj(),Ca.conj(),Ca,Ca)
            h2bb = np.einsum("pqrs,pi,qj,rk,sl->ijkl",h2,Cb.conj(),Cb.conj(),Cb,Cb)
            h2ab = np.einsum("pqrs,pi,qj,rk,sl->ijkl",h2,Ca.conj(),Cb.conj(),Ca,Cb)
            h2_ = np.zeros((Values.NSO,Values.NSO,Values.NSO,Values.NSO),dtype=complex)
            h2_[::2,::2,::2,::2] = h2aa
            h2_[1::2,1::2,1::2,1::2] = h2bb
            h2_[1::2,::2,1::2,::2] = h2ab.transpose(1,0,3,2)
            h2_[::2,1::2,::2,1::2] = h2ab

            # h2_ *= 0
            # h1_ *= 0

            h1_.flatten(order="F").tofile(f"../FCI/{settings.mol_name}_h1.bin")
            h2_.flatten(order="F").tofile(f"../FCI/{settings.mol_name}_h2.bin")
            S.flatten(order="F").tofile(f"../FCI/{settings.mol_name}_S.bin")

        if settings.refine_PHF:
            HOne = np.zeros([Values.NSO, Values.NSO])
            HTwo = np.zeros([Values.NSO, Values.NSO, Values.NSO, Values.NSO])
            HOne[: Values.NAO, : Values.NAO] = Values.h1[:, :]
            HOne[Values.NAO :, Values.NAO :] = Values.h1[:, :]
            HTwo[: Values.NAO, : Values.NAO, : Values.NAO, : Values.NAO] = Values.eri
            HTwo[Values.NAO :, Values.NAO :, Values.NAO :, Values.NAO :] = Values.eri
            HTwo[Values.NAO :, Values.NAO :, : Values.NAO, : Values.NAO] = Values.eri
            HTwo[: Values.NAO, : Values.NAO, Values.NAO :, Values.NAO :] = Values.eri
            HTwo = Mulliken2Dirac(HTwo)
            if settings.DIIS == True:
                Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk = (
                    optPHF_DIIS(HOne, HTwo, Values.SGHFMO, fcomm, settings, Values)
                )
            else:
                Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk = (
                    optPHF_SCF(HOne, HTwo, Values.SGHFMO, fcomm, settings, Values)
                )
            if write_PHF:
                pickle.dump(
                    [Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk],
                    open(f"{write_PHF}", "wb"),
                )
    else:
        z0 = None
        for i, basis in enumerate(settings.basis):
            if i > 0:
                "Up convert basis if not first calculation"
                Values = convert_basis(settings, Values, basis, write_OAO=write_OAO)

            # if i == 1:
            #     # FCI
            #     mci = fci.direct_spin1.FCI()
            #     mci = fci.addons.fix_spin_(mci, ss=settings.J*(settings.J+1))
            #     fcie, fcivec = mci.kernel(Values.h1, Values.eri, Values.NAO, Values.NOccSO, nroots=4)
            #     print("E(FCI1)= %8.8f" %(fcie[0]+Values.Enuc))
            #     print("E(FCI2)= %8.8f" %(fcie[1]+Values.Enuc))
            #     raise

            t1 = time.time()
            "Expand integrals to GHF form and antisymmetrize"
            HOne = np.zeros([Values.NSO, Values.NSO])
            HTwo = np.zeros([Values.NSO, Values.NSO, Values.NSO, Values.NSO])
            HOne[: Values.NAO, : Values.NAO] = Values.h1[:, :]
            HOne[Values.NAO :, Values.NAO :] = Values.h1[:, :]
            HTwo[: Values.NAO, : Values.NAO, : Values.NAO, : Values.NAO] = Values.eri
            HTwo[Values.NAO :, Values.NAO :, Values.NAO :, Values.NAO :] = Values.eri
            HTwo[Values.NAO :, Values.NAO :, : Values.NAO, : Values.NAO] = Values.eri
            HTwo[: Values.NAO, : Values.NAO, Values.NAO :, Values.NAO :] = Values.eri
            HTwo = Mulliken2Dirac(HTwo)

            if step > 0:
                C_old = Values.SGHFMO.copy()

            if i == 0 and step == 0:
            # if True:
                if i > 0:
                    z0, Olap = get_ref(settings, Values, sci=True)

                Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk = optPHF(
                    HOne, HTwo, Values.A2G, settings, Values, i, z0=z0
                )
                # Values.SGHFMO = Values.A2G

                # Finish convergence
                if settings.DIIS == True:
                    Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk = (
                        optPHF_DIIS(HOne, HTwo, Values.SGHFMO, settings, Values)
                    )
                else:
                    Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk = (
                        optPHF_SCF(HOne, HTwo, Values.SGHFMO, settings, Values)
                    )
            else:
                if True:
                    # z0, Olap = get_ref(settings, Values, sci=True)

                    "2026-10-04: pass the real basis index i, not the literal 1."
                    "optPHF treats i > 0 as a basis up-conversion and then forces BFGS, 0 basin"
                    "hops and gtol 1e-3 -- a deliberately loose PRE-optimization that is meant to be"
                    "followed by a tight one in the final basis.  Passing 1 here applied that loose"
                    "setting to every geometry of a scan, where it is the ONLY optimization performed,"
                    "so steps 01.. of an SGHF scan were converged only to max|G| ~ 1e-3 and never saw"
                    "phf_policy (BFGS + >= 3 hops for SP = 1) or the spin-flip kick.  With i (= 0 for a"
                    "single-basis scan step) the documented policy applies at every geometry."
                    Values.SGHFMO = orthonormalize_ref(Values.SGHFMO, Values.NOccSO, settings, step, i)
                    Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk = optPHF(
                        HOne, HTwo, Values.SGHFMO, settings, Values, i, z0=None
                    )
                if settings.DIIS == True:
                    Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk = (
                        optPHF_DIIS(HOne, HTwo, Values.SGHFMO, settings, Values)
                    )
                else:
                    Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk = (
                        optPHF_SCF(HOne, HTwo, Values.SGHFMO, settings, Values)
                    )

            t2 = time.time()
            if settings.VERBOSE > 0:
                "Unprojected energy of the optimised (VAP) reference determinant: <Phi|H|Phi> with the GHF-form integrals"
                Cocc = Values.SGHFMO[:, : Values.NOccSO]; Dm = Cocc @ Cocc.conj().T
                Edet = np.einsum("pq,qp->", HOne, Dm) + 0.5 * np.einsum("pqrs,rp,sq->", HTwo, Dm, Dm) + Values.Enuc
                lab = {2: "UHF", 1: "GHF", 3: "UHF"}.get(settings.SP, "det")
                print(f"E({lab} det, unprojected VAP reference)= {Edet.real:.14f}   E(PHF)= {Values.EPHF + Values.Enuc:.14f}")
            if settings.VERBOSE > 1:
                GHFS = calcS(Values.SGHFMO, Values.NOccSO, Values.NAO)
                GHFSS = calcS2(Values.SGHFMO, Values.NOccSO, Values.NAO)
                print("GHF S^2, Sxyz=", GHFSS, GHFS)
                print("PHF time=", t2 - t1)

            if step > 0 and Values.rank == 0:
                "Rotate orbs to resemble previous step"
                Values.SGHFMO = MatchOrb(Values.SGHFMO, C_old, Values.NOccSO)
                if settings.use_MPI == True:
                    Values.comm.Bcast(Values.SGHFMO, root=0)

            if settings.chkpoint == True and Values.rank == 0:
                pickle.dump(
                    [Values.EPHF, Values.SGHFMO, Values.fsp, Values.fpg, Values.fk],
                    open(f"{write_PHF}", "wb"),
                )
            if ((len(settings.basis) > 1) and (i == 0) and (settings.do_CC == True)):
                "Get cc amps for smallest basis to be projected later"
                Values = run_PCC(settings, Values)
                Values.min_OAO = Values.OAO.copy()

    # n, o = Values.OAO.shape
    # S = np.zeros([2 * n, 2 * o])
    # S[:n, :o] = Values.OAO
    # S[n:, o:] = Values.OAO
    # print((S.dot(Values.SGHFMO))[:,:Values.NOccAO])
    # print((S.dot(Values.SGHFMO))[:,Values.NOccAO:Values.NOccSO])
    if settings.VERBOSE > 0:
        print("E(PHF)=", Values.EPHF + Values.Enuc)
    return Values

    # G2S = Values.A2G.T @ Values.SGHFMO
    # ovlp, Z = GetThouless(G2S)
    # print("ovlp=", ovlp)
    # for i in range(settings.ngrid[0]):
    #     Values.R1[i, :, :] = ao2mo(Values.R1[i, :, :], Values.SGHFMO, 2)
    # for i in range(settings.ngrid[1]):
    #     Values.R2[i, :, :] = ao2mo(Values.R2[i, :, :], Values.SGHFMO, 2)
    # for i in range(Values.npg):
    #     Values.Rpg[i, :, :] = ao2mo(Values.Rpg[i, :, :], Values.SGHFMO, 2)
    # Rk = CmplxProj(Values.SGHFMO, Values.NSO, Values.ncik)
    # print(
    #     "Ovlp=",
    #     EvalOvlp(
    #         Values.R1, Values.R2, Values.Rpg, Rk, Values.fsp, Values.fpg, Values.fk
    #     ),
    # )

    # return Values


def optPHF_SCF(H1, H2, MOs, comm, settings, Values):
    "Run the PHF optimization using SCF procedure for orbitals"

    # if (
    #     (settings.chkpoint == True)
    #     and (settings.read_PHF == True)
    #     and (os.path.exists(f"{settings.mol_name}_PHForb.p"))
    # ):
    #     MOs = pickle.load(open(f"{settings.mol_name}_PHForb.p", "rb"))

    E_old = 76234678342.0

    newMOs = MOs.copy()
    for i in range(settings.PHF_maxiter):
        E0, newMOs, fsp, fpg, fk = update_orbs_old(
            H1, H2, newMOs, comm, settings, Values
        )
        # E0, newMOs, fsp, fpg, fk = update_orbs(H1, H2, newMOs, comm, settings, Values)

        if abs(E0 - E_old) < settings.PHF_thrsh:
            break
        elif i < settings.PHF_maxiter - 1:
            E_old = E0
        else:
            raise Exception("SCF procedure failed to converge")

    if settings.CmplxConj == 2:
        newMOs = FixCmplxGauge(fk[1], newMOs, Values.NOccSO, Values.NSO)
        fk[1] = 1.0
    if settings.fixgauge:
        newMOs = FixGauge(newMOs, Values.NAO, Values.NSO, Values.NOccSO, settings.J)
    return E0, newMOs, fsp, fpg, fk


def update_orbs_old(H1, H2, MOs, comm, settings, Values):
    "Given MOs, build Fock operator and diagonalize for new set of MOs"

    if settings.use_MPI:
        from PHFTools_F import phftools
    else:
        from PHFTools_nompi_F import phftools

    HOne = ao2mo(H1, MOs, 2)
    HTwo = ao2mo(H2, MOs, 4)
    if settings.use_MPI == True:
        Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg_mo(
            HOne,
            HTwo,
            MOs,
            settings.ngrid,
            Values.nci,
            settings.CmplxConj,
            Values.ncisp,
            Values.NOccSO,
            Values.R1,
            Values.R2,
            Values.Rpg,
            Values.weightsp,
            Values.weightpg,
            Values.roota,
            Values.rootb,
            Values.rooty,
            settings.J,
            settings.SP,
            Values.fcomm,
            Values.npg,
            Values.ncipg,
            Values.NSO,
        )
    else:
        Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg_mo(
            HOne,
            HTwo,
            MOs,
            settings.ngrid,
            Values.nci,
            settings.CmplxConj,
            Values.ncisp,
            Values.NOccSO,
            Values.R1,
            Values.R2,
            Values.Rpg,
            Values.weightsp,
            Values.weightpg,
            Values.roota,
            Values.rootb,
            Values.rooty,
            settings.J,
            settings.SP,
            Values.npg,
            Values.ncipg,
            Values.NSO,
        )

    evals, evecs = EigenSolver(Hmat, Smat)
    E0 = evals[0]
    # print("CI Ene=", evals + Values.Enuc)
    fsp, fpg, fk = VecDecomp(evecs[:, 0], Values.ncisp, Values.ncipg, Values.ncik)
    "Construct ov block of Fock"
    G = LocalG(
        Gradmat,
        Rdmmat,
        Smat,
        E0,
        fsp,
        fpg,
        fk,
        Values.ncisp,
        Values.ncipg,
        Values.ncik,
        Values.NOccSO,
        settings.CmplxConj,
    )
    # G = phftools.localg(Gradmat,Rdmmat,Smat,E0,fsp,fpg,fk,CmplxConj,NSO,nci,ncisp,ncipg)

    "Ensure that the correct symmetries are still in place"
    if settings.NFO > 0:
        "Remove rotations with the first NFO occupied orbitals"
        G = FrozenProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccAO, Values.NFO)
    if settings.CmplxConj == 0:
        "Zero out imaginary component of z"
        G.imag = 0
    if settings.SP == 2:
        "Remove alpha beta mixing terms"
        G = SzProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)
    if settings.is_RHF:
        "Make alpha and beta components equal"
        G = RHFProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)

    "Build diagonal blocks of Fock operator from deform determinant"
    GHF = pyscf.scf.GHF(Values.mol)
    GHF.get_hcore = lambda *args: block_diag(Values.h1, Values.h1)
    GHF.get_ovlp = lambda *args: block_diag(Values.Ovlp, Values.Ovlp)
    GHF._eri = pyscf.ao2mo.restore(8, Values.eri, Values.NAO)
    GHF.diis_space = 10
    GHF.level_shift = 0.1
    dm = np.dot(MOs[:, : Values.NOccSO], MOs[:, : Values.NOccSO].T)
    temp = GHF.get_fock(dm=dm)

    F_GHF = ao2mo(temp, MOs, 2)

    "Construct total Fock operator"
    F = np.zeros((Values.NSO, Values.NSO), dtype=complex)
    F[: Values.NOccSO, : Values.NOccSO] = F_GHF[: Values.NOccSO, : Values.NOccSO]
    F[Values.NOccSO :, Values.NOccSO :] = F_GHF[Values.NOccSO :, Values.NOccSO :]
    F[: Values.NOccSO, Values.NOccSO :] = G.T.conj()
    F[Values.NOccSO :, : Values.NOccSO] = G
    if (settings.SP == 2) or (settings.is_RHF == True):
        """
        Separate F into alpha and beta matrices and diagonalize
        separately to ensure that alpha beta mixing does not occur

        orbitals are ordered alpha occ, beta occ, alpha vir, beta vir
        """
        alpha = np.zeros((Values.NSO), dtype=bool)
        alpha[: Values.NOccAO] = 1
        alpha[Values.NOccSO : Values.NOccSO + Values.NVrtA] = 1

        beta = np.invert(alpha)
        # beta = np.zeros((Values.NSO), dtype=bool)
        # beta[Values.NOccAO:Values.NOccSO] = 1
        # beta[Values.NOccSO + Values.NVrtA:] = 1
        Fa = F[:, alpha][alpha, :]
        Fb = F[:, beta][beta, :]
        vala, veca = sl.eigh(Fa)
        valb, vecb = sl.eigh(Fb)
        vec = np.zeros((Values.NSO, Values.NSO), dtype=complex)
        vec[alpha, : Values.NOccAO] = veca[:, : Values.NOccAO]
        vec[beta, Values.NOccAO : Values.NOccSO] = vecb[:, : Values.NOccAO]
        vec[alpha, Values.NOccSO : Values.NOccSO + Values.NVrtA] = veca[
            :, Values.NOccAO :
        ]
        vec[beta, Values.NOccSO + Values.NVrtA :] = vecb[:, Values.NOccAO :]
    else:
        val, vec = sl.eigh(F)

    "Convert newMOs to OAO basis"
    newMOs = MOs.dot(vec)

    output = open("PHFout", "a")
    print(
        "E and max |G|",
        evals[0] + Values.Enuc,
        np.max(np.abs(np.imag(np.diag(Hmat / Smat)))),
        np.max(np.abs(G.real)),
        np.max(np.abs(G.imag)),
        # file=output,
    )
    output.close()
    return E0, newMOs, fsp, fpg, fk


def update_orbs(H1, H2, MOs, comm, settings, Values):
    "Given MOs, build Fock operator and diagonalize for new set of MOs"

    if settings.use_MPI:
        from PHFTools_F import phftools
    else:
        from PHFTools_nompi_F import phftools

    HOne = ao2mo(H1, Values.A2G, 2)
    HTwo = ao2mo(H2, Values.A2G, 4)
    z, Olap = GetThoulessZ(Values.A2G, MOs, Values.NOccSO)
    if settings.use_MPI == True:
        raise Exception("MPI not implemented for SCF yet")
    else:
        Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg(
            HOne,
            HTwo,
            Values.A2G,
            z,
            settings.ngrid,
            Values.nci,
            settings.CmplxConj,
            Values.ncisp,
            Values.NOccSO,
            Values.R1,
            Values.R2,
            Values.Rpg,
            Values.weightsp,
            Values.weightpg,
            Values.roota,
            Values.rootb,
            Values.rooty,
            settings.J,
            settings.SP,
            Values.npg,
            Values.ncipg,
            Values.NSO,
        )

    evals, evecs = EigenSolver(Hmat, Smat)
    E0 = evals[0]
    # print("CI Ene=", evals + Values.Enuc)
    fsp, fpg, fk = VecDecomp(evecs[:, 0], Values.ncisp, Values.ncipg, Values.ncik)
    "Construct ov block of Fock"
    G = LocalG(
        Gradmat,
        Rdmmat,
        Smat,
        E0,
        fsp,
        fpg,
        fk,
        Values.ncisp,
        Values.ncipg,
        Values.ncik,
        Values.NOccSO,
        settings.CmplxConj,
    )
    # G = phftools.localg(Gradmat,Rdmmat,Smat,E0,fsp,fpg,fk,CmplxConj,NSO,nci,ncisp,ncipg)

    # L = sl.cholesky(np.eye(Values.NOccSO) + (z.T).dot(z.conj()), lower=True)
    # L = sl.solve_triangular(L, np.eye(Values.NOccSO), lower=True)
    # M = sl.cholesky(np.eye(Values.NVrtSO) + z.conj().dot(z.T), lower=True)
    # M = sl.solve_triangular(M, np.eye(Values.NVrtSO), lower=True)
    # print(M)
    # print(G[:,:5])
    # print((M.T.dot(G).dot(L.conj()))[:,:5])
    # raise

    "Ensure that the correct symmetries are still in place"
    if settings.NFO > 0:
        "Remove rotations with the first NFO occupied orbitals"
        G = FrozenProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccAO, Values.NFO)
    if settings.CmplxConj == 0:
        "Zero out imaginary component of z"
        G.imag = 0
    if settings.SP == 2:
        "Remove alpha beta mixing terms"
        G = SzProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)
    if settings.is_RHF:
        "Make alpha and beta components equal"
        G = RHFProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)

    "Build diagonal blocks of Fock operator from deform determinant"
    GHF = pyscf.scf.GHF(Values.mol)
    GHF.get_hcore = lambda *args: block_diag(Values.h1, Values.h1)
    GHF.get_ovlp = lambda *args: block_diag(Values.Ovlp, Values.Ovlp)
    GHF._eri = pyscf.ao2mo.restore(8, Values.eri, Values.NAO)
    GHF.diis_space = 10
    GHF.level_shift = 0.1
    dm = np.dot(MOs[:, : Values.NOccSO], MOs[:, : Values.NOccSO].T)
    temp = GHF.get_fock(dm=dm)

    # F_GHF = temp.copy()
    # temp = ao2mo(temp, Values.A2G, 2)
    F_GHF = ao2mo(temp, MOs, 2)
    # F_GHF = ao2mo(temp, Values.A2G, 2)
    # print(G[:, : 5])
    # raise

    "Construct total Fock operator"
    F = np.zeros((Values.NSO, Values.NSO), dtype=complex)
    F[: Values.NOccSO, : Values.NOccSO] = F_GHF[: Values.NOccSO, : Values.NOccSO]
    F[Values.NOccSO :, Values.NOccSO :] = F_GHF[Values.NOccSO :, Values.NOccSO :]
    F[: Values.NOccSO, Values.NOccSO :] = G.T.conj()
    F[Values.NOccSO :, : Values.NOccSO] = G
    if (settings.SP == 2) or (settings.is_RHF == True):
        """
        Separate F into alpha and beta matrices and diagonalize
        separately to ensure that alpha beta mixing does not occur

        orbitals are ordered alpha occ, beta occ, alpha vir, beta vir
        """
        alpha = np.zeros((Values.NSO), dtype=bool)
        alpha[: Values.NOccAO] = 1
        alpha[Values.NOccSO : Values.NOccSO + Values.NVrtA] = 1

        beta = np.invert(alpha)
        # beta = np.zeros((Values.NSO), dtype=bool)
        # beta[Values.NOccAO:Values.NOccSO] = 1
        # beta[Values.NOccSO + Values.NVrtA:] = 1
        Fa = F[:, alpha][alpha, :]
        Fb = F[:, beta][beta, :]
        vala, veca = sl.eigh(Fa)
        valb, vecb = sl.eigh(Fb)
        vec = np.zeros((Values.NSO, Values.NSO), dtype=complex)
        vec[alpha, : Values.NOccAO] = veca[:, : Values.NOccAO]
        vec[beta, Values.NOccAO : Values.NOccSO] = vecb[:, : Values.NOccAO]
        vec[alpha, Values.NOccSO : Values.NOccSO + Values.NVrtA] = veca[
            :, Values.NOccAO :
        ]
        vec[beta, Values.NOccSO + Values.NVrtA :] = vecb[:, Values.NOccAO :]
    else:
        val, vec = sl.eigh(F)

    "Convert newMOs to OAO basis"
    # print(vec)
    # raise
    newMOs = MOs.dot(vec)
    # newMOs = Values.A2G.dot(vec)
    # print(vala)
    # print(valb)
    # raise

    output = open("PHFout", "a")
    print(
        "E and max |G|",
        evals[0] + Values.Enuc,
        np.max(np.abs(np.imag(np.diag(Hmat / Smat)))),
        np.max(np.abs(G.real)),
        np.max(np.abs(G.imag)),
        # file=output,
    )
    output.close()
    return E0, newMOs, fsp, fpg, fk


def optPHF_DIIS(H1, H2, MOs, settings, Values):
    """
    Use DIIS to optimize PHF orbitals. Error vectors are occupied/virtual
    block of commutator of Fock operator and density.

    """

    if settings.VERBOSE > 2:
        print("Using DIIS to get PHF orbitals")

    status = False

    "Prepare DIIS(Need first two steps for method to work)"

    "Build initial density"
    den = np.einsum(
        "ij,kj->ik",
        MOs[:, : Values.NOccSO],
        MOs[:, : Values.NOccSO].conj(),
        optimize=True,
    )

    "Get Fock"
    F, E0, fsp, fpg, fk = build_F(H1, H2, den, settings, Values)
    e = (F.dot(den) - den.dot(F)).flatten()
    if settings.VERBOSE > 2:
        print(f"Initial PHF energy and error: {E0 + Values.Enuc}", sl.norm(e))

    Fock = []
    Fock.append(F)

    "For the error vectors"
    err = []

    "Initialize DIIS matrix and vector"
    B = -1 * np.ones((3, 3))
    B[2, 2] = 0
    v = np.array([0, 0, -1])

    "Add first term"
    Bii = np.vdot(e, e)
    err.append(e)
    B[0, 0] = np.real(Bii)

    "Check if initial guess converged"
    if np.sqrt(Bii) < settings.PHF_thrsh:
        if settings.VERBOSE > 1:
            print("Initial guess already converged")
        return E0, MOs, fsp, fpg, fk

    "Build second step by diagonalization"

    "Diagonalize F"
    if (settings.SP == 2) or (settings.is_RHF == True):
        """
        Separate F into alpha and beta matrices and diagonalize
        separately to ensure that alpha beta mixing does not occur

        OAO orbitals are ordered alpha, beta
        """
        alpha = np.zeros((Values.NSO), dtype=bool)
        alpha[: Values.NAO] = 1

        beta = np.invert(alpha)

        Fa = F[:, alpha][alpha, :]
        Fb = F[:, beta][beta, :]
        vala, veca = sl.eigh(Fa)
        valb, vecb = sl.eigh(Fb)
        newMOs = np.zeros((Values.NSO, Values.NSO), dtype=complex)
        newMOs[alpha, : Values.NOccAO] = veca[:, : Values.NOccAO]
        newMOs[beta, Values.NOccAO : Values.NOccSO] = vecb[:, : Values.NOccAO]
        newMOs[alpha, Values.NOccSO : Values.NOccSO + Values.NVrtA] = veca[
            :, Values.NOccAO :
        ]
        newMOs[beta, Values.NOccSO + Values.NVrtA :] = vecb[:, Values.NOccAO :]
    else:
        val, newMOs = sl.eigh(F)

    den = np.einsum(
        "ij,kj->ik",
        newMOs[:, : Values.NOccSO],
        newMOs[:, : Values.NOccSO].conj(),
        optimize=True,
    )

    "Do DIIS"

    "Number of states being used for DIIS"
    m = 2

    for n in range(1, settings.PHF_maxiter):
        if status == True:
            break

        if settings.VERBOSE > 2:
            print(f"PHF opt step {n}")

        "Get Fock"
        F, E0, fsp, fpg, fk = build_F(H1, H2, den, settings, Values)
        Fock.append(F)
        e = (F.dot(den) - den.dot(F)).flatten()

        "Update DIIS matrix"
        Bii = np.vdot(e, e)
        err.append(e)
        B[-2, -2] = np.real(Bii)

        for j in range(m - 1):
            "Fill in new errors"
            e = np.vdot(err[j], err[-1])
            B[j, -2] = B[-2, j] = np.real(e)

        "If condition number of B is too small, use eigenvector of F for new guess instead"
        if np.linalg.cond(B, p=-2) < 1.0e-15:
            F = Fock[-1]

            "Check for convergence, use diagonal value for error"
            error = Bii

        else:
            "Get weights"
            w = sl.solve(B, v, assume_a="sym")
            w = w[:-1]
            if settings.use_MPI == True:
                Values.comm.Bcast(w, root=0)
            error = w.dot(B[:-1, :-1]).dot(w)

            "Make new Fock"
            F_ = np.zeros_like(Fock[-1])
            for j in range(m):
                F_ += w[j] * Fock[j]
            F = F_.copy()

        if (settings.SP == 2) or (settings.is_RHF == True):
            """
            Separate F into alpha and beta matrices and diagonalize
            separately to ensure that alpha beta mixing does not occur

            OAO orbitals are ordered alpha, beta
            """
            alpha = np.zeros((Values.NSO), dtype=bool)
            alpha[: Values.NAO] = 1

            beta = np.invert(alpha)

            Fa = F[:, alpha][alpha, :]
            Fb = F[:, beta][beta, :]
            vala, veca = sl.eigh(Fa)
            valb, vecb = sl.eigh(Fb)
            newMOs = np.zeros((Values.NSO, Values.NSO), dtype=complex)
            newMOs[alpha, : Values.NOccAO] = veca[:, : Values.NOccAO]
            newMOs[beta, Values.NOccAO : Values.NOccSO] = vecb[:, : Values.NOccAO]
            newMOs[alpha, Values.NOccSO : Values.NOccSO + Values.NVrtA] = veca[
                :, Values.NOccAO :
            ]
            newMOs[beta, Values.NOccSO + Values.NVrtA :] = vecb[:, Values.NOccAO :]
        else:
            val, newMOs = sl.eigh(F)

        if settings.use_MPI == True:
            Values.comm.Bcast(newMOs, root=0)

        den = np.einsum(
            "ij,kj->ik",
            newMOs[:, : Values.NOccSO],
            newMOs[:, : Values.NOccSO].conj(),
            optimize=True,
        )

        if settings.VERBOSE > 2:
            print(f"Current PHF energy and error: {E0 + Values.Enuc}", np.sqrt(error))
        if settings.VERBOSE > 4:
            print("Current Fock:")
            print(F[-1])
            print("Current density:")
            print(den)

        "Check for convergence"
        if np.sqrt(error) < settings.PHF_thrsh:
            status = True
            break
        elif n == settings.PHF_maxiter - 1:
            print("Failed to converge state")
            raise

        """
        Remove states whose error is significatly higher than latest, ensuring
        that we keep at least two
        """
        d = 0
        for i in range(m - 1):
            if B[i, i] > 1000 * B[-2, -2]:
                m -= 1
                d += 1
                del err[i]
                del Fock[i]

        m += 1

        "Expand DIIS matrix and vector"
        B_ = -1 * np.ones((m + 1, m + 1))
        B_[-1, -1] = 0.0
        B_[:-2, :-2] = B[d:-1, d:-1]
        B = B_.copy()
        v = np.zeros((m + 1))
        v[-1] = -1.0

    "Diagonalize F for return"
    F, E0, fsp, fpg, fk = build_F(H1, H2, den, settings, Values)
    if (settings.SP == 2) or (settings.is_RHF == True):
        """
        Separate F into alpha and beta matrices and diagonalize
        separately to ensure that alpha beta mixing does not occur

        OAO orbitals are ordered alpha, beta
        """
        alpha = np.zeros((Values.NSO), dtype=bool)
        alpha[: Values.NAO] = 1

        beta = np.invert(alpha)

        Fa = F[:, alpha][alpha, :]
        Fb = F[:, beta][beta, :]
        vala, veca = sl.eigh(Fa)
        valb, vecb = sl.eigh(Fb)
        newMOs = np.zeros((Values.NSO, Values.NSO), dtype=complex)
        newMOs[alpha, : Values.NOccAO] = veca[:, : Values.NOccAO]
        newMOs[beta, Values.NOccAO : Values.NOccSO] = vecb[:, : Values.NOccAO]
        newMOs[alpha, Values.NOccSO : Values.NOccSO + Values.NVrtA] = veca[
            :, Values.NOccAO :
        ]
        newMOs[beta, Values.NOccSO + Values.NVrtA :] = vecb[:, Values.NOccAO :]
    else:
        val, newMOs = sl.eigh(F)

    if settings.use_MPI == True:
        Values.comm.Bcast(newMOs, root=0)

    if settings.CmplxConj == 2:
        if Values.rank == 0:
            newMOs = FixCmplxGauge(fk[1], newMOs, Values.NOccSO, Values.NSO)
        fk[1] = 1.0
        if settings.use_MPI == True:
            Values.comm.Bcast(newMOs, root=0)
    if settings.fixgauge:
        if Values.rank == 0:
            newMOs = FixGauge(newMOs, Values.NAO, Values.NSO, Values.NOccSO, settings.J)
        if settings.use_MPI == True:
            Values.comm.Bcast(newMOs, root=0)

    if status:
        return E0, newMOs, fsp, fpg, fk


def build_F(H1, H2, den, settings, Values):
    """
    Build the Fock operator from the deform density in the OAO basis

    H1, H2, and den should be in the OAO basis
    """

    if settings.use_MPI:
        from PHFTools_F import phftools
    else:
        from PHFTools_nompi_F import phftools

    "Get MOs from density"
    if (settings.SP == 2) or (settings.is_RHF == True):
        """
        Separate den into alpha and beta matrices and diagonalize
        separately to ensure that alpha beta mixing does not occur
        """
        alpha = np.zeros((Values.NSO), dtype=bool)
        alpha[: Values.NAO] = 1

        beta = np.invert(alpha)

        dena = den[:, alpha][alpha, :]
        denb = den[:, beta][beta, :]
        vala, veca = sl.eigh(-dena)
        valb, vecb = sl.eigh(-denb)
        MOs = np.zeros((Values.NSO, Values.NSO), dtype=complex)
        MOs[alpha, : Values.NOccAO] = veca[:, : Values.NOccAO]
        MOs[beta, Values.NOccAO : Values.NOccSO] = vecb[:, : Values.NOccAO]
        MOs[alpha, Values.NOccSO : Values.NOccSO + Values.NVrtA] = -veca[
            :, Values.NOccAO :
        ]
        MOs[beta, Values.NOccSO + Values.NVrtA :] = -vecb[:, Values.NOccAO :]
    else:
        val, MOs = sl.eigh(-den)

    if settings.use_MPI == True:
        Values.comm.Bcast(MOs, root=0)

    # HOne = ao2mo(H1, Values.A2G, 2)
    # HTwo = ao2mo(H2, Values.A2G, 4)
    # z, Olap = GetThoulessZ(Values.A2G,MOs,Values.NOccSO)
    # if settings.use_MPI == True:
    #     raise Exception("MPI not implemented for SCF yet")
    # else:
    #     Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg(
    #         HOne,
    #         HTwo,
    #         Values.A2G,
    #         z,
    #         settings.ngrid,
    #         Values.nci,
    #         settings.CmplxConj,
    #         Values.ncisp,
    #         Values.NOccSO,
    #         Values.R1,
    #         Values.R2,
    #         Values.Rpg,
    #         Values.weightsp,
    #         Values.weightpg,
    #         Values.roota,
    #         Values.rootb,
    #         Values.rooty,
    #         settings.J,
    #         settings.SP,
    #         Values.npg,
    #         Values.ncipg,
    #         Values.NSO,
    #     )
    HOne = ao2mo(H1, MOs, 2)
    HTwo = ao2mo(H2, MOs, 4)
    if settings.use_MPI == True:
        Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg_mo(
            HOne,
            HTwo,
            MOs,
            settings.ngrid,
            Values.nci,
            settings.CmplxConj,
            Values.ncisp,
            Values.NOccSO,
            Values.R1,
            Values.R2,
            Values.Rpg,
            Values.weightsp,
            Values.weightpg,
            Values.roota,
            Values.rootb,
            Values.rooty,
            settings.J,
            settings.SP,
            Values.fcomm,
            Values.npg,
            Values.ncipg,
            Values.NSO,
        )
    else:
        Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg_mo(
            HOne,
            HTwo,
            MOs,
            settings.ngrid,
            Values.nci,
            settings.CmplxConj,
            Values.ncisp,
            Values.NOccSO,
            Values.R1,
            Values.R2,
            Values.Rpg,
            Values.weightsp,
            Values.weightpg,
            Values.roota,
            Values.rootb,
            Values.rooty,
            settings.J,
            settings.SP,
            Values.npg,
            Values.ncipg,
            Values.NSO,
        )

    evals, evecs = EigenSolver(Hmat, Smat)
    E0 = evals[0]
    # print("CI Ene=", evals + Values.Enuc)
    fsp, fpg, fk = VecDecomp(evecs[:, 0], Values.ncisp, Values.ncipg, Values.ncik)
    "Construct ov block of Fock"
    G = LocalG(
        Gradmat,
        Rdmmat,
        Smat,
        E0,
        fsp,
        fpg,
        fk,
        Values.ncisp,
        Values.ncipg,
        Values.ncik,
        Values.NOccSO,
        settings.CmplxConj,
    )
    # G = phftools.localg(Gradmat,Rdmmat,Smat,E0,fsp,fpg,fk,CmplxConj,NSO,nci,ncisp,ncipg)

    "Ensure that the correct symmetries are still in place"
    if settings.NFO > 0:
        "Remove rotations with the first NFO occupied orbitals"
        G = FrozenProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccAO, Values.NFO)
    if settings.CmplxConj == 0:
        "Zero out imaginary component of z"
        G.imag = 0
    if settings.SP == 2:
        "Remove alpha beta mixing terms"
        G = SzProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)
    if settings.is_RHF:
        "Make alpha and beta components equal"
        G = RHFProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)

    "Build diagonal blocks of Fock operator from deform determinant"
    GHF = pyscf.scf.GHF(Values.mol)
    GHF.get_hcore = lambda *args: block_diag(Values.h1, Values.h1)
    GHF.get_ovlp = lambda *args: block_diag(Values.Ovlp, Values.Ovlp)
    GHF._eri = pyscf.ao2mo.restore(8, Values.eri, Values.NAO)
    GHF.diis_space = 10
    GHF.level_shift = 0.1
    temp = GHF.get_fock(dm=den)
    F_GHF = ao2mo(temp, MOs, 2)
    # F_GHF = ao2mo(temp, Values.A2G, 2)

    "Construct total Fock operator"
    F = np.zeros((Values.NSO, Values.NSO), dtype=complex)
    F[: Values.NOccSO, : Values.NOccSO] = F_GHF[: Values.NOccSO, : Values.NOccSO]
    F[Values.NOccSO :, Values.NOccSO :] = F_GHF[Values.NOccSO :, Values.NOccSO :]
    F[: Values.NOccSO, Values.NOccSO :] = G.T.conj()
    F[Values.NOccSO :, : Values.NOccSO] = G

    if settings.use_MPI == True:
        Values.comm.Bcast(F, root=0)

    "Convert F from NO to OAO basis"
    # F = ao2mo(F, Values.A2G.T, 2)
    F = ao2mo(F, MOs.T, 2)

    return F, E0, fsp, fpg, fk


ZBOUND = 5.0   # bound on Thouless parameters for IPOPT; secondary protection only (validation is primary)


def orthonormalize_ref(MOs, NOccSO, settings, step, i, tol=1.0e-10):
    """Check, and if necessary restore, orthonormality of a warm-start determinant.

    Between geometries the working one-particle basis (Values.OAO = the RHF orbitals at that
    geometry, aligned to the previous step by MatchOrb) is rebuilt, and the previous step's
    determinant is reinterpreted in it.  Both bases are orthonormal and MatchOrb is a unitary
    (orthogonal-Procrustes) rotation, so C stays unitary in exact arithmetic; this guard catches
    the cases where it does not -- accumulated round-off along a long chain, a rank-deficient
    MatchOrb virtual block, or a determinant that came from a pickle written by another run.
    Restoration is a Lowdin symmetric orthonormalization, which is the minimal change: it is the
    unitary matrix closest to C in the Frobenius norm, so it perturbs the determinant as little
    as possible while making it a legitimate starting point.
    """
    dev = float(np.abs(MOs.conj().T @ MOs - np.eye(MOs.shape[1])).max())
    if dev <= tol:
        if settings.VERBOSE > 1:
            print(f"warm start (step {step}, basis {i}): reference orthonormal to {dev:.1e}")
        return MOs
    u, sv, vh = np.linalg.svd(MOs)
    MOnew = u @ vh
    dev2 = float(np.abs(MOnew.conj().T @ MOnew - np.eye(MOnew.shape[1])).max())
    if settings.VERBOSE > 0:
        print(f"warm start (step {step}, basis {i}): reference NOT orthonormal (max|C^dag C - 1| = {dev:.3e}); "
              f"Lowdin-restored to {dev2:.1e}, singular values {sv.min():.6f}..{sv.max():.6f}, "
              f"max|dC| {float(np.abs(MOnew - MOs).max()):.3e}")
    if sv.min() < 1e-8:
        raise ValueError(f"warm-start reference is rank deficient (smallest singular value {sv.min():.3e}); "
                         "the determinant cannot be transferred to this geometry")
    return MOnew


def phf_policy(settings):
    """Optimizer policy -> (use_ipopt, number of basin hops).  See PHF_opt in PCC_objects."""
    if settings.PHF_opt == "ipopt":
        return True, settings.PHF_basinhop
    if settings.PHF_opt == "bfgs":
        return False, settings.PHF_basinhop
    if settings.SP == 2:                                   # auto: SUHF -> IPOPT
        return True, settings.PHF_basinhop
    if settings.SP == 1:                                   # auto: SGHF -> BFGS + >= 3 hops (robust mode)
        return False, max(settings.PHF_basinhop, 3)
    return False, settings.PHF_basinhop


def collinear_reference(MOs, Values, tol=1e-8):
    """True if every occupied orbital of the reference is pure alpha or pure beta (Sz eigenstate)."""
    occ = MOs[:, : Values.NOccSO]
    na = np.sum(np.abs(occ[: Values.NAO]) ** 2, axis=0)
    nb = np.sum(np.abs(occ[Values.NAO :]) ** 2, axis=0)
    return bool(np.all(np.minimum(na, nb) < tol))


def sghf_initial_kick(z0, settings, Values, MOs):
    """SP = 1: if the reference is collinear, perturb the spin-flip block of z0 deterministically so the
    optimization never starts exactly on the collinear (Sz-symmetric) stationary manifold.
    Returns (z0, kicked)."""
    if settings.SP != 1 or settings.PHF_sghf_kick <= 0 or not collinear_reference(MOs, Values):
        return z0, False
    nv, no = Values.NVrtSO, Values.NOccSO
    flip = ((np.arange(nv)[:, None] < Values.NVrtA) != (np.arange(no)[None, :] < Values.NOccA)).reshape(-1)
    idx = np.where(flip)[0]                                # real parts of the spin-flip block
    rng = np.random.default_rng(20260928)                  # fixed seed: reproducible
    z = np.array(z0, dtype=float, copy=True)
    z[idx] += settings.PHF_sghf_kick * rng.standard_normal(len(idx))
    return z, True


def phf_optimize(z0, eg_args, settings, Values, use_ipopt, nhop, gtol):
    """Run the local optimizer (IPOPT or BFGS), optionally inside scipy basinhopping."""
    if use_ipopt:
        if minimize_ipopt is None:
            raise ImportError('PHF_opt = "ipopt" but cyipopt is not installed; use PHF_opt = "bfgs"')
        ipopt_opts = {
            "tol": gtol,
            "max_iter": 50 * settings.PHF_maxiter,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "none",
            "hessian_approximation": "limited-memory",
            "print_level": 5 if settings.VERBOSE > 2 else 0,
        }

        def local_min(fun, x0, args=(), jac=None, **unused):
            "IPOPT as a scipy-compatible local minimizer; scipy.minimize passes fun (scalar) and jac separately"
            return minimize_ipopt(fun, x0, jac=(jac if callable(jac) else True), args=args,
                                  bounds=[(-ZBOUND, ZBOUND)] * len(x0), options=ipopt_opts)
        minimizer_kwargs = {"method": local_min, "jac": True, "args": eg_args}
    else:
        minimizer_kwargs = {"method": "BFGS", "jac": True, "args": eg_args, "options": {"gtol": gtol}}
    if nhop > 0:
        return basinhopping(EandG, z0, minimizer_kwargs=minimizer_kwargs, niter=nhop,
                            T=settings.PHF_hop_T, stepsize=settings.PHF_hop_step)
    return minimize(EandG, z0, **minimizer_kwargs)


def validate_phf_result(res, E_start, eg_args, gtol, label, Enuc, verbose=1):
    """Independent post-solve validation of ANY optimizer result.  Recomputes E and G with EandG and
    requires: finite parameters, energy and gradient; max|G| <= 10*gtol; energy not above the
    starting energy; parameters inside the bound.  Raises RuntimeError otherwise.  The optimizer's
    own success/status flag is deliberately not used.  Returns (E_fin, gmax)."""
    x = np.asarray(res.x, dtype=float)
    problems = []
    E_fin, gmax, zmax = float("nan"), float("nan"), float(np.max(np.abs(x))) if x.size else 0.0
    if not np.all(np.isfinite(x)):
        problems.append("nonfinite parameters")
    else:
        E_fin, G_fin = EandG(x, *eg_args)
        gmax = float(np.max(np.abs(G_fin)))
        if not (np.isfinite(E_fin) and np.all(np.isfinite(G_fin))):
            problems.append("nonfinite energy or gradient")
        else:
            if E_fin > E_start + 1e-10:
                problems.append(f"energy rose: start {E_start + Enuc:.10f} -> final {E_fin + Enuc:.10f}")
            if gmax > 10.0 * gtol:
                problems.append(f"max|G| = {gmax:.2e} > 10 * tolerance {10*gtol:.1e}")
        if zmax > 0.99 * ZBOUND:
            problems.append(f"parameters diverged: max|z| = {zmax:.3f} (bound {ZBOUND})")
    if verbose > 0:
        print(f"PHF optimizer: {label}: E = {E_fin + Enuc:.10f}, max|G| = {gmax:.2e}, max|z| = {zmax:.3f}, "
              f"nfev = {getattr(res, 'nfev', '?')}")
    if problems:
        raise RuntimeError(f"PHF result ({label}) rejected by post-solve validation: " + "; ".join(problems))
    return E_fin, gmax


def optPHF(HOne, HTwo, MOs, settings, Values, i, z0=None):
    "Run the PHF optimization using direct minimization of orbitals"

    if settings.use_MPI:
        from PHFTools_F import phftools
    else:
        from PHFTools_nompi_F import phftools

    "Convert to the natural orbital basis"
    H1 = ao2mo(HOne, MOs, 2)
    H2 = ao2mo(HTwo, MOs, 4)
    # Sx = BuildSx(Values.NAO)
    # Sy = BuildSy(Values.NAO)
    # Sz = BuildSz(Values.NAO)
    Zeros = np.zeros([Values.NSO, Values.NSO, Values.NSO, Values.NSO])
    if type(z0) == type(None):
        # if (
        #     (settings.chkpoint == True)
        #     and (settings.read_PHF == True)
        #     and (os.path.exists(f"{settings.mol_name}_PHFzmat.p"))
        # ):
        #     z0 = pickle.load(open(f"{settings.mol_name}_PHFzmat.p", "rb"))
        # else:
        z0 = np.zeros(2 * (Values.NSO - Values.NOccSO) * Values.NOccSO)

    eg_args = (H1, H2, MOs, settings, Values, phftools)
    if i == 0:
        use_ipopt, nhop = phf_policy(settings)
        gtol = settings.PHF_ipopt_tol if use_ipopt else settings.PHF_bfgs_gtol
        z0, kicked = sghf_initial_kick(z0, settings, Values, MOs)
        if kicked and settings.VERBOSE > 0:
            print(f"SGHF: collinear reference detected -> spin-flip kick of {settings.PHF_sghf_kick:.1e} applied to z0")
        label = ("IPOPT" if use_ipopt else "BFGS") + (f" + {nhop} basin hop(s)" if nhop > 0 else "")
    else:
        use_ipopt, nhop, gtol, label = False, 0, 1.0e-3, "BFGS (basis up-conversion)"
    E_start = EandG(z0, *eg_args)[0]
    res = phf_optimize(z0, eg_args, settings, Values, use_ipopt, nhop, gtol)
    validate_phf_result(res, E_start, eg_args, gtol, label, Values.Enuc, settings.VERBOSE)
    res.success = True          # validation above is the authority, not the optimizer's own flag
    if settings.VERBOSE > 1:
        print(res)
    if res.success == False:
        raise Exception("PHF optimization failed to converge!")
    sol = (
        res.x[: Values.NVrtSO * Values.NOccSO]
        + res.x[Values.NVrtSO * Values.NOccSO :] * 1j
    )
    if settings.use_MPI == True:
        Values.comm.Bcast(sol, root=0)

    # if settings.chkpoint == True:
    #     pickle.dump(res.x, open(f"{settings.mol_name}_PHFzmat.p", "wb"))
    z = sol.reshape([Values.NSO - Values.NOccSO, Values.NOccSO])
    if settings.NFO > 0:
        "Remove rotations with the first NFO occupied orbitals"
        z = FrozenProj(z, Values.NVrtSO, Values.NOccSO, Values.NOccAO, Values.NFO)
    if settings.CmplxConj == 0:
        "Zero out imaginary component of z"
        z.imag = 0
    if settings.SP == 2:
        "Remove alpha beta mixing terms"
        z = SzProj(z, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)
    if settings.is_RHF:
        "Make alpha and beta components equal"
        z = RHFProj(z, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)
    G2S = Thouless2MOs(z, Values.NSO, Values.NOccSO)
    newMOs = MOs.dot(G2S)
    if settings.fixgauge:
        if Values.rank == 0:
            newMOs = FixGauge(newMOs, Values.NAO, Values.NSO, Values.NOccSO, settings.J)
        if settings.use_MPI == True:
            Values.comm.Bcast(newMOs, root=0)
    if settings.use_MPI == True:
        Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg(
            H1,
            H2,
            MOs,
            z,
            settings.ngrid,
            Values.nci,
            settings.CmplxConj,
            Values.ncisp,
            Values.NOccSO,
            Values.R1,
            Values.R2,
            Values.Rpg,
            Values.weightsp,
            Values.weightpg,
            Values.roota,
            Values.rootb,
            Values.rooty,
            settings.J,
            settings.SP,
            Values.fcomm,
            Values.npg,
            Values.ncipg,
            Values.NSO,
        )
    else:
        Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg(
            H1,
            H2,
            MOs,
            z,
            settings.ngrid,
            Values.nci,
            settings.CmplxConj,
            Values.ncisp,
            Values.NOccSO,
            Values.R1,
            Values.R2,
            Values.Rpg,
            Values.weightsp,
            Values.weightpg,
            Values.roota,
            Values.rootb,
            Values.rooty,
            settings.J,
            settings.SP,
            Values.npg,
            Values.ncipg,
            Values.NSO,
        )
    evals, evecs = EigenSolver(Hmat, Smat)
    E0 = evals[0]
    fsp, fpg, fk = VecDecomp(evecs[:, 0], Values.ncisp, Values.ncipg, Values.ncik)
    print("CI Ene=", evals + Values.Enuc)
    if settings.CmplxConj == 2:
        if Values.rank == 0:
            newMOs = FixCmplxGauge(fk[1], newMOs, Values.NOccSO, Values.NSO)
        fk[1] = 1.0
        if settings.use_MPI == True:
            Values.comm.Bcast(newMOs, root=0)
    return E0, newMOs, fsp, fpg, fk


def EandG(Z, *args):
    HOne, HTwo, MOs, settings, Values, phftools = args
    lamb = 0
    if settings.use_MPI == True:
        Values.comm.Bcast(Z, root=0)

    # if settings.chkpoint == True:
    #     pickle.dump(Z, open(f"{settings.mol_name}_PHFzmat.p", "wb"))
    z0 = Z[: Values.NVrtSO * Values.NOccSO] + Z[Values.NVrtSO * Values.NOccSO :] * 1j
    sol = z0.reshape(Values.NSO - Values.NOccSO, Values.NOccSO)
    if settings.NFO > 0:
        "Remove rotations with the first NFO occupied orbitals"
        sol = FrozenProj(sol, Values.NVrtSO, Values.NOccSO, Values.NOccAO, Values.NFO)
    if settings.CmplxConj == 0:
        "Zero out imaginary component of z"
        sol.imag = 0
    if settings.SP == 2:
        "Remove alpha beta mixing terms"
        sol = SzProj(sol, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)
    if settings.is_RHF:
        "Make alpha and beta components equal"
        sol = RHFProj(sol, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)
    # Hmat, Smat, Gradmat, Rdmmat = BuildHSG(HOne,HTwo,MOs,sol)
    if settings.use_MPI == True:
        Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg(
            HOne,
            HTwo,
            MOs,
            sol,
            settings.ngrid,
            Values.nci,
            settings.CmplxConj,
            Values.ncisp,
            Values.NOccSO,
            Values.R1,
            Values.R2,
            Values.Rpg,
            Values.weightsp,
            Values.weightpg,
            Values.roota,
            Values.rootb,
            Values.rooty,
            settings.J,
            settings.SP,
            Values.fcomm,
            Values.npg,
            Values.ncipg,
            Values.NSO,
        )
    else:
        Hmat, Smat, Gradmat, Rdmmat = phftools.buildhsg(
            HOne,
            HTwo,
            MOs,
            sol,
            settings.ngrid,
            Values.nci,
            settings.CmplxConj,
            Values.ncisp,
            Values.NOccSO,
            Values.R1,
            Values.R2,
            Values.Rpg,
            Values.weightsp,
            Values.weightpg,
            Values.roota,
            Values.rootb,
            Values.rooty,
            settings.J,
            settings.SP,
            Values.npg,
            Values.ncipg,
            Values.NSO,
        )
    evals, evecs = EigenSolver(Hmat, Smat)
    E0 = evals[0]
    fsp, fpg, fk = VecDecomp(evecs[:, 0], Values.ncisp, Values.ncipg, Values.ncik)
    G = LocalG(
        Gradmat,
        Rdmmat,
        Smat,
        E0,
        fsp,
        fpg,
        fk,
        Values.ncisp,
        Values.ncipg,
        Values.ncik,
        Values.NOccSO,
        settings.CmplxConj,
    )
    # G = phftools.localg(Gradmat,Rdmmat,Smat,E0,fsp,fpg,fk,CmplxConj,NSO,nci,ncisp,ncipg)
    E0 = evals[0] + lamb * z0.T.conj() @ z0

    G = G + lamb * z0.reshape(Values.NSO - Values.NOccSO, Values.NOccSO)
    if settings.NFO > 0:
        "Remove rotations with the first NFO occupied orbitals"
        G = FrozenProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccAO, Values.NFO)
    if settings.CmplxConj == 0:
        "Zero out imaginary component of z"
        G.imag = 0
    if settings.SP == 2:
        "Remove alpha beta mixing terms"
        G = SzProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)
    if settings.is_RHF:
        "Make alpha and beta components equal"
        G = RHFProj(G, Values.NVrtSO, Values.NOccSO, Values.NOccA, Values.NVrtA)
    G = G.reshape((Values.NSO - Values.NOccSO) * Values.NOccSO)
    # output = open("PHFout", "a")
    if Values.rank == 0:
        print(
            "E and max |G|",
            evals[0] + Values.Enuc,
            np.max(np.abs(np.imag(np.diag(Hmat / Smat)))),
            np.max(np.abs(G.real)),
            np.max(np.abs(G.imag)),
            # file=output,
        )
    # output.close()
    # G = dE/dz* (Wirtinger); for the real parameters x = Re z, y = Im z of a real E:
    # dE/dx = 2 Re G, dE/dy = 2 Im G.  (Without the 2 the gradient is half the true one --
    # harmless for scipy BFGS, but it makes IPOPT's line search fail; verified with
    # IPOPT's derivative checker: analytic/FD = 0.5000 on every component.)
    return E0.real, 2.0 * np.concatenate((G.real, G.imag))


def SzCons(Z, MO):
    z0 = Z[: NVrtSO * NOccSO] + Z[NVrtSO * NOccSO :] * 1j
    z0 = z0.reshape(NSO - NOccSO, NOccSO)
    if CmplxConj == 0:
        z0.imag = 0
    sz = ao2mo(Sz, MO, 2)
    s_z = MFEne(sz, Zeros, z0)
    return s_z.real


def SzConsGrad(Z, MO):
    z0 = Z[: NVrtSO * NOccSO] + Z[NVrtSO * NOccSO :] * 1j
    z0 = z0.reshape(NSO - NOccSO, NOccSO)
    if CmplxConj == 0:
        z0.imag = 0
    sz = ao2mo(Sz, MO, 2)
    s_z = MFEne(sz, Zeros, z0)
    szg = MFGrad(sz, Zeros, z0, s_z)
    print(
        "Sz and max |SG|", s_z.real, np.max(np.abs(szg.real)), np.max(np.abs(szg.imag))
    )
    return np.concatenate((szg.real, szg.imag))


def SxCons(Z, MO):
    z0 = Z[: NVrtSO * NOccSO] + Z[NVrtSO * NOccSO :] * 1j
    z0 = z0.reshape(NSO - NOccSO, NOccSO)
    if CmplxConj == 0:
        z0.imag = 0
    sx = ao2mo(Sx, MO, 2)
    s_x = MFEne(sx, Zeros, z0)
    return s_x.real


def SxConsGrad(Z, MO):
    z0 = Z[: NVrtSO * NOccSO] + Z[NVrtSO * NOccSO :] * 1j
    z0 = z0.reshape(NSO - NOccSO, NOccSO)
    if CmplxConj == 0:
        z0.imag = 0
    sx = ao2mo(Sx, MO, 2)
    s_x = MFEne(sx, Zeros, z0)
    sxg = MFGrad(sx, Zeros, z0, s_x)
    print(
        "Sx and max |SG|", s_x.real, np.max(np.abs(sxg.real)), np.max(np.abs(sxg.imag))
    )
    return np.concatenate((sxg.real, sxg.imag))


def SzProj(Z, NVrtSO, NOccSO, NOccA, NVrtA):
    "Remove alpha beta mixing terms"
    Znew = np.zeros([NVrtSO, NOccSO], dtype=complex)
    Znew[:NVrtA, :NOccA] = Z[:NVrtA, :NOccA]
    Znew[NVrtA:, NOccA:] = Z[NVrtA:, NOccA:]
    return Znew


def RHFProj(Z, NVrtSO, NOccSO, NOccA, NVrtA):
    "Define beta Z matrix to be identical to alpha"
    Znew = np.zeros([NVrtSO, NOccSO], dtype=complex)
    Znew[:NVrtA, :NOccA] = Z[:NVrtA, :NOccA]
    Znew[NVrtA:, NOccA:] = Z[:NVrtA, :NOccA]
    return Znew


# The input MO should be RHF type
def FrozenProj(Z, NVrtSO, NOccSO, NOccAO, NFO):
    "Zero rotations with the first NFO orbitals"
    Znew = np.zeros([NVrtSO, NOccSO], dtype=complex)
    Znew[:, :] = Z
    Znew[:, :NFO] = 0
    Znew[:, NOccAO : NOccAO + NFO] = 0
    return Znew


def PseudoDiag(HOne, HTwo, A2G):
    from makeH import MakeFock

    newMO = np.zeros_like(A2G)
    F = MakeFock(HOne, HTwo, A2G[:, :NOccSO])
    e, V = np.linalg.eigh(F)
    uocc = V[:, :NOccSO].T.conj() @ A2G[:, :NOccSO]
    R = subspace_eigh(np.diag(e[:NOccSO]), uocc)[1]
    newMO[:, :NOccSO] = V[:, :NOccSO] @ R
    uvir = V[:, NOccSO:].T.conj() @ A2G[:, NOccSO:]
    R = subspace_eigh(np.diag(e[NOccSO:]), uvir)[1]
    newMO[:, NOccSO:] = V[:, NOccSO:] @ R
    return newMO


def subspace_eigh(Fock, MO_Occ):
    f = MO_Occ.T.conj() @ Fock @ MO_Occ
    moe, u = np.linalg.eigh(f)
    return moe, MO_Occ @ u
