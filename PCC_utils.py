import numpy as np
import scipy.linalg as sl
import sys
from copy import deepcopy
from scipy.optimize import minimize
from ao2mo import ao2mo
from solve_ci import solve_ci, Lanczos, Davidson

def FrozenCore(H1, H2, MO, NFC, NFV, NAO):
    # use rRHF MO energy as reference
    # E0 = 2 * <i|h|i> + 2 * [ii|jj] - [ij|ji], h2 is in Mulliken order
    # <p|h|q>  => <p|h|q> + 2 [ii|pq] - [pi|iq]
    HOne = ao2mo(H1, MO, 2)
    HTwo = ao2mo(H2, MO, 4)
    NMO = NAO - NFC - NFV
    E0 = 0
    H1new = np.zeros([NMO, NMO])
    H2new = np.zeros([NMO, NMO, NMO, NMO])
    for i in range(NFC):
        E0 += 2 * HOne[i, i]
        for j in range(NFC):
            E0 += 2 * HTwo[i, i, j, j] - HTwo[i, j, j, i]
    for p in range(NFC, NAO - NFV):
        for q in range(NFC, NAO - NFV):
            H1new[p - NFC, q - NFC] += HOne[p, q]
            for i in range(NFC):
                H1new[p - NFC, q - NFC] += 2 * HTwo[i, i, p, q] - HTwo[p, i, i, q]
    for p in range(NFC, NAO - NFV):
        for q in range(NFC, NAO - NFV):
            for r in range(NFC, NAO - NFV):
                for s in range(NFC, NAO - NFV):
                    H2new[p - NFC, q - NFC, r - NFC, s - NFC] = HTwo[p, q, r, s]
    return E0, H1new, H2new


def FixGauge(MOs, NAO, NSO, NOccSO, J):
    S2One, S2Two = BuildS2(NAO)
    S1 = ao2mo(S2One, MOs, 2)
    S2 = ao2mo(S2Two, MOs, 4)
    guess = np.zeros(3)
    res = minimize(SandG, guess, args=(S1, S2, MOs, NAO, NSO, NOccSO, J), method="BFGS", jac=True)
    # 	a,b,ib,y,iy = res.x
    a, ib, y = res.x
    b = 0
    iy = 0
    A = complex(a, 0)
    B = complex(b, ib)
    Y = complex(y, iy)
    R = BuildRealRotMat(A, B, Y, NAO, NSO)
    R = ao2mo(R, MOs, 2)
    ovlp, Z = GetThouless(R, NOccSO)
    newMOs = Thouless2MOs(Z, NSO, NOccSO)
    return MOs @ newMOs


def SandG(x0, *args):
    SOne, STwo, MOs, NAO, NSO, NOccSO, J = args
    SS = SVal(x0, SOne, STwo, MOs, NAO, NSO, NOccSO)
    glist = np.zeros([3, 2])
    dx = 1e-3
    s2 = J * (J + 1)
    for n in range(len(glist[0, :])):
        for i in range(len(glist[:, 0])):
            xp = deepcopy(x0)
            xn = deepcopy(x0)
            xp[i] += (n + 1) * dx
            xn[i] -= (n + 1) * dx
            fp = SVal(xp, SOne, STwo, MOs, NAO, NSO, NOccSO)
            fn = SVal(xn, SOne, STwo, MOs, NAO, NSO, NOccSO)
            glist[i, n] = (fp - fn) / (2 * (n + 1) * dx)
    Sg = (4 * glist[:, 0] - glist[:, 1]) / 3
    print("S and max |G|", SS - s2, np.max(np.abs(Sg)))
    return (SS - s2) ** 2, 2 * (SS - s2) * Sg


def SVal(x0, SOne, STwo, MOs, NAO, NSO, NOccSO):
    # 	a,b,ib,y,iy = x0
    a, ib, y = x0
    b = 0
    iy = 0
    A = complex(a, 0)
    B = complex(b, ib)
    Y = complex(y, iy)
    R = BuildRealRotMat(A, B, Y, NAO, NSO)
    R = ao2mo(R, MOs, 2)
    ovlp, Z = GetThouless(R, NOccSO)
    SS = MFEne(SOne, 4 * STwo, Z, NSO, NOccSO)
    return SS.real


def BuildRealRotMat(a, b, y, NAO, NSO):
    Ap = np.exp(a / 2)
    An = np.exp(-a / 2)
    Bp = (np.exp(b / 2) + np.exp(-b / 2)) / 2
    Bn = (np.exp(b / 2) - np.exp(-b / 2)) / 2j
    Yp = np.exp(y / 2)
    Yn = np.exp(-y / 2)
    SaRotMat = np.zeros([NSO, NSO], dtype=complex)
    SbRotMat = np.zeros([NSO, NSO], dtype=complex)
    SyRotMat = np.zeros([NSO, NSO], dtype=complex)
    for i in range(NAO):
        SaRotMat[i][i] = Ap
        SaRotMat[i + NAO][i + NAO] = An
        SbRotMat[i][i] = SbRotMat[i + NAO][i + NAO] = Bp
        SbRotMat[i][i + NAO] = Bn
        SbRotMat[i + NAO][i] = -Bn
        SyRotMat[i][i] = Yp
        SyRotMat[i + NAO][i + NAO] = Yn
    RotMat = SaRotMat @ SbRotMat @ SyRotMat
    return RotMat


def FixCmplxGauge(fk, MO, NOccSO, NSO):
    phi = -np.angle(fk) / (2 * NOccSO)
    newMO = np.exp(1j * phi) * np.eye(NSO)
    return newMO @ MO


def BuildSz(NAO):
    "Construct Sz operator in spin orbital basis"
    NSO = 2 * NAO
    Sz = np.zeros([NSO, NSO])
    for i in range(NAO):
        Sz[i, i] = 1
        Sz[i + NAO, i + NAO] = -1
    return Sz / 2


def BuildSy(NAO):
    "Construct Sy operator in spin orbital basis"
    NSO = 2 * NAO
    Sy = np.zeros([NSO, NSO], dtype=complex)
    for i in range(NAO):
        Sy[i, i + NAO] = -1j
        Sy[i + NAO, i] = 1j
    return Sy / 2


def BuildSx(NAO):
    "Construct Sx operator in spin orbital basis"
    NSO = 2 * NAO
    Sx = np.zeros([NSO, NSO])
    for i in range(NAO):
        Sx[i, i + NAO] = 1
        Sx[i + NAO, i] = 1
    return Sx / 2


def BuildS2(NAO):
    NSO = NAO * 2
    S2One = np.zeros([NSO, NSO])
    S2Two = np.zeros([NSO, NSO, NSO, NSO])
    for i in range(NAO):
        S2One[i, i] = 0.75
        S2One[i + NAO, i + NAO] = 0.75
    for i in range(NAO):
        for j in range(NAO):
            S2Two[i + NAO, j + NAO, i + NAO, j + NAO] = 0.25
            S2Two[i, j, i, j] = 0.25
            S2Two[i, j + NAO, i + NAO, j] = 1
            S2Two[i, j + NAO, i, j + NAO] = -0.5
    S2Two = antisymm(S2Two)
    return S2One, S2Two

def antisymm(H2):
	tmp = H2 - np.einsum('ijkl->ijlk',H2)
	H2new = tmp - np.einsum('ijkl->jikl',tmp)
	return H2new / 4

def calcS(Ref, NOcc, NAO):
    NSO = 2 * NAO
    Sx = BuildSx(NAO)
    Sy = BuildSz(NAO)
    Sz = BuildSz(NAO)
    rdm = np.matmul(Ref[:, :NOcc], Ref[:, :NOcc].T.conj())
    ExpSx = np.einsum("ij,ji", Sx, rdm)
    ExpSy = np.einsum("ij,ji", Sy, rdm)
    ExpSz = np.einsum("ij,ji", Sz, rdm)
    return ExpSx, ExpSy, ExpSz


def calcS2(Ref, NOcc, NAO):
    NSO = NAO * 2
    S2One, S2Two = BuildS2(NAO)
    rdm = np.matmul(Ref[:, :NOcc], Ref[:, :NOcc].T.conj())
    ExpS2 = np.einsum("ij,ji", S2One, rdm)
    ExpS2 += np.einsum("ijkl,ki,lj", S2Two, rdm, rdm)
    ExpS2 += -np.einsum("ijkl,li,kj", S2Two, rdm, rdm)
    return ExpS2


def SpinDen(MO):
    mo = orth @ MO
    mo1 = mo[:NAO, :NAO]
    mo2 = mo[NAO:, NAO:]
    Sa = mo1[:, :NOccAO] @ mo1[:, :NOccAO].T.conj()
    Sb = mo2[:, :NOccAO] @ mo2[:, :NOccAO].T.conj()
    return np.diag(Sa - Sb)


def SSHG(MO, NAO, NOccSO):
    Sx = BuildSx(NAO)
    Sy = BuildSy(NAO)
    Sz = BuildSz(NAO)
    Ox = MO.T.conj() @ Sx @ MO
    Oy = MO.T.conj() @ Sy @ MO
    Oz = MO.T.conj() @ Sz @ MO
    Olist = [Ox, Oy, Oz]
    A = np.eye(3, dtype=complex) * NOccSO / 4
    for i in range(3):
        for j in range(3):
            A[i, j] -= np.trace(Olist[i] @ Olist[j])
    return A


def EigenSolver(H, S, Tol=1.e-6):
    evals, evecs = np.linalg.eigh(S)
    Dim = len(evals)
    invS = [1 / np.sqrt(abs(i)) for i in evals if (np.abs(i) > Tol)]
    dim = len(invS)
    Sinv = np.diag(invS)
    X = np.matmul(evecs[:, Dim - dim :], Sinv)
    h = X.T.conj() @ H @ X
    evals, evecs = np.linalg.eigh(h)
    evecs = X @ evecs
    return evals, evecs


def VecDecomp(vec, ncisp, ncipg, ncik):
    M = vec.reshape((ncisp * ncipg, ncik), order="F")
    fk = M[0, :] / M[0, 0]
    M = M[:, 0].reshape((ncisp, ncipg), order="F")
    fpg = M[0, :] / M[0, 0]
    fsp = M[:, 0]
    return fsp, fpg, fk


def EvalOvlp(r1, r2, rpg, rk, fsp, fpg, fk):
    Ovlp = 0
    newMOs = np.identity(NSO, dtype=complex)
    for il in range(ngrid[0]):
        for iy in range(ngrid[1]):
            for ipg in range(npg):
                for ik in range(ncik):
                    R = r1[il, :, :] @ r2[iy, :, :] @ rpg[ipg, :, :] @ rk[ik, :, :]
                    rho0, s0 = Rdm(R, newMOs[:, :NOccSO])
                    s0 = weightsp[il, iy] * s0
                    for ni in range(ncisp):
                        for nj in range(ncisp):
                            if SP == 2:
                                wig = WignerMat(
                                    J, ni - J, nj - J, roota[il], rootb[iy], rooty[il]
                                )
                            else:
                                wig = WignerMat(
                                    J, ni - J, nj - J, roota[il], rootb[il], rooty[iy]
                                )
                            for p in range(ncipg):
                                for q in range(ncipg):
                                    wpg = weightpg[ipg, p, q]
                                    w = (
                                        wig
                                        * wpg
                                        * fsp[ni].conj()
                                        * fsp[nj]
                                        * fpg[p].conj()
                                        * fpg[q]
                                        * fk[ik]
                                    )
                                    Ovlp += s0 * w
    return Ovlp


def LocalG(
    Gradmat, Rdmmat, Smat, E0, fsp, fpg, fk, ncisp, ncipg, ncik, NOccSO, CmplxConj
):
    G = 0
    Ovlp = 0
    Rdm = 0
    for ni in range(ncisp):
        for nj in range(ncisp):
            for p in range(ncipg):
                for q in range(ncipg):
                    m = ni + p * ncisp
                    n = nj + q * ncisp
                    G += (
                        fsp[ni].conj()
                        * fsp[nj]
                        * fpg[p].conj()
                        * fpg[q]
                        * fk[0].conj()
                        * fk[0]
                        * Gradmat[m, n, :, :]
                    )
                    Ovlp += (
                        fsp[ni].conj()
                        * fsp[nj]
                        * fpg[p].conj()
                        * fpg[q]
                        * fk[0].conj()
                        * fk[0]
                        * Smat[m, n]
                    )
                    Rdm += (
                        fsp[ni].conj()
                        * fsp[nj]
                        * fpg[p].conj()
                        * fpg[q]
                        * fk[0].conj()
                        * fk[0]
                        * Rdmmat[m, n, :, :]
                    )
                    if CmplxConj == 2:
                        d = ncisp * ncipg
                        G += (
                            fsp[ni].conj()
                            * fsp[nj]
                            * fpg[p].conj()
                            * fpg[q]
                            * fk[1].conj()
                            * fk[1]
                            * Gradmat[m + d, n + d, :, :].conj()
                        )
                        Ovlp += (
                            fsp[ni].conj()
                            * fsp[nj]
                            * fpg[p].conj()
                            * fpg[q]
                            * fk[1].conj()
                            * fk[1]
                            * Smat[m + d, n + d]
                        )
                        Rdm += (
                            fsp[ni].conj()
                            * fsp[nj]
                            * fpg[p].conj()
                            * fpg[q]
                            * fk[1].conj()
                            * fk[1]
                            * Rdmmat[m + d, n + d, :, :].conj()
                        )
                        G += (
                            fsp[ni].conj()
                            * fsp[nj]
                            * fpg[p].conj()
                            * fpg[q]
                            * fk[0].conj()
                            * fk[1]
                            * Gradmat[m, n + d, :, :]
                        )
                        Ovlp += (
                            fsp[ni].conj()
                            * fsp[nj]
                            * fpg[p].conj()
                            * fpg[q]
                            * fk[0].conj()
                            * fk[1]
                            * Smat[m, n + d]
                        )
                        Rdm += (
                            fsp[ni].conj()
                            * fsp[nj]
                            * fpg[p].conj()
                            * fpg[q]
                            * fk[0].conj()
                            * fk[1]
                            * Rdmmat[m, n + d, :, :]
                        )
                        G += (
                            fsp[ni].conj()
                            * fsp[nj]
                            * fpg[p].conj()
                            * fpg[q]
                            * fk[0].conj()
                            * fk[1]
                            * Gradmat[m + d, n, :, :].conj()
                        )
                        Ovlp += (
                            fsp[ni].conj()
                            * fsp[nj]
                            * fpg[p].conj()
                            * fpg[q]
                            * fk[1].conj()
                            * fk[0]
                            * Smat[m + d, n]
                        )
                        Rdm += (
                            fsp[ni].conj()
                            * fsp[nj]
                            * fpg[p].conj()
                            * fpg[q]
                            * fk[0].conj()
                            * fk[1]
                            * Rdmmat[m + d, n, :, :].conj()
                        )
    G -= E0 * Rdm

    return G[NOccSO:, :NOccSO] / Ovlp
    # G[NOccSO:,:NOccSO] /= Ovlp
    # return G

def Rdm(R, MOs):
    M = np.matmul(MOs.T.conj(), R)
    M = np.matmul(M, MOs)
    Minv = np.linalg.inv(M)
    det = np.linalg.det(M)
    rdm = np.matmul(R, MOs)
    rdm = np.matmul(rdm, Minv)
    rdm = np.matmul(rdm, MOs.T.conj())
    return rdm, det


def RdmConj(R, MOs):
    M = np.matmul(MOs.T.conj(), R)
    M = np.matmul(M, MOs.conj())
    Minv = np.linalg.inv(M)
    det = np.linalg.det(M)
    rdm = np.matmul(R, MOs.conj())
    rdm = np.matmul(rdm, Minv)
    rdm = np.matmul(rdm, MOs.T.conj())
    return rdm, det


def Kernels(HOne, HTwo, rdm, det):
    h = np.einsum("ij,ji", HOne, rdm)
    h += 0.5 * np.einsum("ijkl,ki,lj", HTwo, rdm, rdm)
    h = det * h
    return h


def Gradient(HOne, HTwo, rho, det):
    h = Makeh(HOne, HTwo, rho)
    gamma = MakeGamma(HTwo, rho)
    g = h * rho
    g += (Id - rho) @ (HOne + gamma) @ rho
    g = g * det
    return g


def MakeGamma(HTwo, rho):
    return np.einsum("ijkl,lj->ik", HTwo, rho)


def Makeh(HOne, HTwo, rho):
    h = np.trace(HOne @ rho)
    gamma = MakeGamma(HTwo, rho)
    h += 0.5 * np.trace(gamma @ rho)
    return h


def MLdecompose(Z, NSO, NOccSO):
    Id = np.identity(NOccSO)
    mat = Id + Z.T @ Z.conj()
    L = np.linalg.cholesky(mat)
    Id = np.identity(NSO - NOccSO)
    mat = Id + Z.conj() @ Z.T
    M = np.linalg.cholesky(mat)
    return L, M


def SortOrb(MO, NAO, NOccAO, NSO, NOccSO, Type):
    # if type = 1
    # rearrange orbitals from OA VA  0  0 to OA  0 VA  0
    #                          0  0 OB VB     0 OB  0 VB
    # if type = 0, do the inverse
    temp = MO.dtype.str
    if Type == 1:
        NVrtAO = NAO - NOccAO
        newMO = np.zeros([NSO, NSO],dtype=temp)
        newMO[:, :NOccAO] = MO[:, :NOccAO]
        newMO[:, NOccAO:NOccSO] = MO[:, NAO : NAO + NOccAO]
        newMO[:, NOccSO : NOccSO + NVrtAO] = MO[:, NOccAO : NOccAO + NVrtAO]
        newMO[:, NAO + NOccAO : NSO] = MO[:, NAO + NOccAO : NSO]
        return newMO
    elif Type == 0:
        NVrtA = NAO - NOccAO
        newMO = np.zeros([NSO, NSO],dtype=temp)
        newMO[:, :NOccAO] = MO[:, :NOccAO]
        newMO[:, NOccAO:NAO] = MO[:, NOccSO : NOccSO + NVrtA]
        newMO[:, NAO : NAO + NOccAO] = MO[:, NOccAO:NOccSO]
        newMO[:, NAO + NOccAO : NSO] = MO[:, NAO + NOccAO : NSO]
        return newMO


def makeUHF(Z):
    # for parameter Zai, zero out the spin-flip part
    newZ = Z
    newZ[NVrtA:, :NOccA] = 0
    newZ[:NVrtA, NOccA:] = 0
    return newZ


def MFEne(HOne, HTwo, Z, NSO, NOccSO):
    newMOs = np.eye(NSO, NOccSO, dtype=complex)
    newMOs[NOccSO:, :NOccSO] = Z
    rdm, ovlp = Rdm(np.eye(NSO), newMOs)
    h = Kernels(HOne, HTwo, rdm, ovlp)
    return h / ovlp


def MFGrad(HOne, HTwo, Z, E0):
    Id = np.identity(NSO)
    newMOs = np.eye(NSO, NOccSO, dtype=complex)
    newMOs[NOccSO:, :NOccSO] = Z
    rdm, ovlp = Rdm(np.eye(NSO), newMOs)
    gk = Gradient(HOne, HTwo, rdm, ovlp) - E0 * rdm * ovlp
    return gk[NOccSO:, :NOccSO] / ovlp


def GetThouless(MO, NOccSO):
    M = MO[:NOccSO, :NOccSO]
    ovlp = np.linalg.det(M)
    Minv = np.linalg.inv(M)
    Z = MO[NOccSO:, :NOccSO] @ Minv
    return ovlp, Z

def GetThoulessZ(MO1, MO2, NOccSO):
    """
    Find Thouless rotation that connects MO1 and MO2
    |2> = e^Z|1>*Olap
    """
    M = MO1.T.conj().dot(MO2)
    L = M[:NOccSO, :NOccSO]
    Y = M[NOccSO:, :NOccSO]
    Linv = sl.inv(L)
    Z = Y.dot(Linv)
    Olap = np.linalg.det(L)

    return Z, Olap


def Thouless2MOs(Z, NSO, NOccSO):
    MOs = np.identity(NSO, dtype=complex)
    MOs[NOccSO:, :NOccSO] = Z
    MOs[:NOccSO, NOccSO:] = -Z.T.conj()
    L, M = MLdecompose(Z, NSO, NOccSO)
    Linv = np.linalg.inv(L)
    Minv = np.linalg.inv(M)
    newMOs = np.zeros([NSO, NSO], dtype=complex)
    newMOs[:, :NOccSO] = MOs[:, :NOccSO] @ Linv.T
    newMOs[:, NOccSO:] = MOs[:, NOccSO:] @ Minv.T
    return newMOs


def AddFrozenMO(MO, NFO):
    newMO = np.zeros([NSO + NFO * 2, NSO + NFO * 2])
    newMO[:NFO, :NFO] = np.eye(NFO)
    newMO[NFO : NFO + NAO, NFO : NFO + NOccA] = MO[:NAO, :NOccA]
    newMO[NFO * 2 + NAO :, NFO : NFO + NOccA] = MO[NAO:, :NOccA]
    newMO[NAO + NFO : NAO + 2 * NFO, NOccA + NFO : NOccA + 2 * NFO] = np.eye(NFO)
    newMO[NFO : NFO + NAO, NOccA + 2 * NFO : NOccSO + 2 * NFO] = MO[:NAO, NOccA:NOccSO]
    newMO[NFO * 2 + NAO :, NOccA + 2 * NFO : NOccSO + 2 * NFO] = MO[NAO:, NOccA:NOccSO]
    newMO[NFO : NFO + NAO, NOccSO + 2 * NFO :] = MO[:NAO, NOccSO:]
    newMO[NFO * 2 + NAO :, NOccSO + 2 * NFO :] = MO[NAO:, NOccSO:]
    return newMO


def RmvFrozenMO(MO, NFO):
    newMO = np.zeros([NSO - NFO * 2, NSO - NFO * 2])
    newMO[: NAO - NFO, : NOccA - NFO] = MO[NFO:NAO, NFO:NOccA]
    newMO[NAO - NFO :, : NOccA - NFO] = MO[NFO + NAO : NSO, NFO:NOccA]
    newMO[: NAO - NFO, NOccA - NFO : NOccSO - 2 * NFO] = MO[
        NFO:NAO, NOccA + NFO : NOccSO
    ]
    newMO[NAO - NFO :, NOccA - NFO : NOccSO - 2 * NFO] = MO[
        NFO + NAO : NSO, NOccA + NFO : NOccSO
    ]
    newMO[: NAO - NFO, NOccSO - 2 * NFO :] = MO[NFO:NAO, NOccSO:]
    newMO[NAO - NFO :, NOccSO - 2 * NFO :] = MO[NFO + NAO : NSO, NOccSO:]
    return newMO
