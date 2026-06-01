import pickle
import time
import os
import pyscf
from scipy.linalg import block_diag
from Cmplx_Proj import CmplxProj
from makeH import SemiCanon, Mulliken2Dirac
from ao2mo import ao2mo
import numpy as np
from MatchOrb import MatchOrb


def ASymm2(T2):
    "Antisymmetrize double excitation operator"
    T2 -= T2.transpose(0, 1, 3, 2)
    T2 -= T2.transpose(1, 0, 2, 3)
    return T2


def CC2CI(T0, T1, T2):
    """
    Convert CC amplitudes to CI amplitudes
    """
    NVir, NOcc = T1.shape
    C0 = np.exp(T0)
    C1 = C0 * T1
    temp = np.einsum("ai,bj->abij", T1, T1, optimize=True)
    for i in range(NOcc):
        temp[:, :, i, i] *= 0
    for a in range(NVir):
        temp[a, a, :, :] *= 0
    C2 = 0.5 * temp
    C2 += 0.25 * T2
    C2 *= C0
    C2 = ASymm2(C2)
    C2 /= 4

    return C0, C1, C2


def CI2CC(C0, C1, C2):
    """
    Convert CI amplitudes to CC amplitudes
    """
    NVir, NOcc = C1.shape
    T0 = np.log(C0)
    T1 = C1 / C0
    T2 = C2 / C0
    temp = np.einsum("ai,bj->abij", T1, T1, optimize=True)
    for i in range(NOcc):
        temp[:, :, i, i] *= 0
    for a in range(NVir):
        temp[a, a, :, :] *= 0
    T2 -= 0.5 * temp
    T2 *= C0
    T2 = ASymm2(T2)

    return T0, T1, T2


def convert_amp_basis(settings, Values, old_basis, old_OAO, old_orb):
    """
    Given a set of amplitudes from old basis, project up to current basis.
    Projection is done by converting CC amplitudes to CI amplitudes in SD
    space, projecting CI wavefunction, then converting back to CC amplitudes in
    new basis.

    To make calculation faster, orbitals in new basis are rotated to look like
    old orbitals making the orbital overlap matrix strongly diagonal. This
    allows us to correspond determinants of old basis with those in the new
    basis and limit the projection to the single element that matches each
    element of the smaller basis CI vector.
    """

    "Build old mol object"
    old_mol = pyscf.gto.Mole()
    old_mol.atom = Values.mol.atom
    old_mol.unit = Values.mol.unit
    old_mol.spin = Values.mol.spin
    old_mol.basis = old_basis
    old_mol.build()

    "Get overlap between old and new basis then convert to RHF basis on each side"
    S_cross = pyscf.gto.mole.intor_cross("int1e_ovlp", Values.mol, old_mol)
    S_cross = Values.OAO.T.dot(S_cross).dot(old_OAO)

    "Expand overlap to GHF form"
    n, o = S_cross.shape
    S = np.zeros([2 * n, 2 * o])
    S[:n, :o] = S_cross[:, :]
    S[n:, o:] = S_cross[:, :]

    Values.semi_MO = MatchOrb(Values.semi_MO, S.dot(old_orb), Values.NOccSO)
    Overlap = np.diag(Values.semi_MO.T.conj().dot(S).dot(old_orb))

    C0, C1, C2 = CC2CI(0.0, Values.PCC[2], Values.PCC[3])

    C1_new = np.zeros((Values.NVrtSO, Values.NOccSO), dtype=complex)
    C2_new = np.zeros(
        (Values.NVrtSO, Values.NVrtSO, Values.NOccSO, Values.NOccSO), dtype=complex
    )
    norm = np.prod(Overlap[: Values.NOccSO])
    vir, occ = Values.PCC[2].shape

    for i in range(occ):
        for a in range(vir):
            C1_new[a, i] = C1[a, i] * norm * Overlap[a] / Overlap[i]
            for j in range(occ):
                for b in range(vir):
                    temp = norm * Overlap[a] * Overlap[b] / Overlap[i] / Overlap[j]
                    C2_new[a, b, i, j] = C2[a, b, i, j] * temp
                    C2_new[b, a, i, j] = C2[b, a, i, j] * temp
                    C2_new[a, b, j, i] = C2[a, b, j, i] * temp
                    C2_new[b, a, j, i] = C2[b, a, j, i] * temp

    _, T1_new, T2_new = CI2CC(1.0, C1_new, C2_new)

    return Values, T1_new, T2_new


def run_PCC(settings, Values, step=0, convert=False, read_PCC=None, write_PCC=None):
    "Driver for PCC calculation"
    if settings.use_MPI:
        from PUCC_X2 import pgcc
    else:
        from PUCC_X2_nompi import pgcc

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
    if ((step > 0) or (convert == True)):
        temp = Values.semi_MO.copy()
    Values.semi_MO = SemiCanon(HOne, HTwo, Values.SGHFMO, Values.NOccSO, settings.SP)
    if step > 0:
        Values.semi_MO = MatchOrb(Values.semi_MO, temp, Values.NOccSO)

    if read_PCC:
        if settings.VERBOSE > 1:
            print(f"Reading PCC values from {read_PCC}")
        E, S, T1_in, T2_in = pickle.load(open(f"{read_PCC}", "rb"))
    elif convert == True:
        Values, T1_in, T2_in = convert_amp_basis(settings, Values, settings.basis[0], Values.min_OAO, temp)
        if settings.use_MPI == True:
            Values.comm.Bcast(T1_in, root=0)
            Values.comm.Bcast(T2_in, root=0)
    elif step > 0:
        T1_in = Values.PCC[2].copy()
        T2_in = Values.PCC[3].copy()
    else:
        T1_in = np.zeros((Values.NVrtSO, Values.NOccSO), dtype=complex)
        T2_in = np.zeros(
            (Values.NVrtSO, Values.NVrtSO, Values.NOccSO, Values.NOccSO), dtype=complex
        )
    if settings.use_MPI == True:
        Values.comm.Bcast(Values.semi_MO, root=0)

    # PCC
    H1 = ao2mo(HOne, Values.semi_MO, 2)
    H2 = ao2mo(HTwo, Values.semi_MO, 4)
    HOne = None
    HTwo = None
    R1 = np.zeros_like(Values.R1)
    R2 = np.zeros_like(Values.R2)
    Rpg = np.zeros_like(Values.Rpg)
    for i in range(settings.ngrid[0]):
        R1[i, :, :] = ao2mo(Values.R1[i, :, :], Values.semi_MO, 2)
    for i in range(settings.ngrid[1]):
        R2[i, :, :] = ao2mo(Values.R2[i, :, :], Values.semi_MO, 2)
    for i in range(Values.npg):
        Rpg[i, :, :] = ao2mo(Values.Rpg[i, :, :], Values.semi_MO, 2)
    Rk = CmplxProj(Values.semi_MO, Values.NSO, Values.ncik)
    # print("Ovlp=", EvalOvlp(R1,R2,Rpg,Rk,fsp,fpg,fk))
    if Values.ncik == 1:
        Values.fk = np.ones(1)

    if settings.use_MPI == True:
        Values.PCC = pgcc(
            Values.semi_MO,
            H1,
            H2,
            T1_in,
            T2_in,
            Values.NAO,
            Values.NOccSO,
            settings.J,
            settings.CmplxConj,
            settings.SP,
            settings.ngrid,
            R1,
            R2,
            Rpg,
            Rk,
            Values.roota,
            Values.rootb,
            Values.rooty,
            Values.weightsp,
            Values.weightpg,
            Values.fsp,
            Values.fpg,
            Values.fk,
            settings.nBroyVec,
            X,
            Xinv,
            settings.chkpoint,
            settings.read_PCC,
            settings.mol_name,
            Values.Enuc,
            Values.fcomm,
            Values.NSO,
            Values.npg,
            Values.ncisp,
            Values.ncipg,
            Values.ncik,
        )
    else:
        Values.PCC = pgcc(
            Values.semi_MO,
            H1,
            H2,
            T1_in,
            T2_in,
            Values.NAO,
            Values.NOccSO,
            settings.J,
            settings.CmplxConj,
            settings.SP,
            settings.ngrid,
            R1,
            R2,
            Rpg,
            Rk,
            Values.roota,
            Values.rootb,
            Values.rooty,
            Values.weightsp,
            Values.weightpg,
            Values.fsp,
            Values.fpg,
            Values.fk,
            settings.nBroyVec,
            X,
            Xinv,
            settings.chkpoint,
            settings.read_PCC,
            settings.mol_name,
            Values.Enuc,
            Values.NSO,
            Values.npg,
            Values.ncisp,
            Values.ncipg,
            Values.ncik,
        )
    print("E(PCC)=", Values.PCC[0].real)
    # print(PCC[1])
    # print(PCC[2])
    # print(PCC[3])
    if write_PCC and Values.rank == 0:
        pickle.dump(
            [Values.PCC[0], Values.PCC[1], Values.PCC[2], Values.PCC[3], Values.semi_MO],
            open(f"{write_PCC}", "wb"),
        )

    return Values
