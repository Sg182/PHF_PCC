import numpy as np

from PCC_utils import SortOrb
import pickle
import time
import os
import pyscf
from pyscf import gto, scf, fci, cc, tdscf
from scipy.linalg import block_diag


def prepare_orbs(settings, Values):
    """
    Run HF calculations to produce the set of starting orbitals for PHF.
    If is_RHF is true, then use RHF. If SP == 2, use UHF. Otherwise use GHF.
    """

    if settings.orb_init == None:
        ## RHF
        for n in range(10):
            RHF = scf.RHF(Values.mol)
            RHF.get_hcore = lambda *args: Values.h1
            RHF.get_ovlp = lambda *args: Values.Ovlp
            RHF._eri = pyscf.ao2mo.restore(8, Values.eri, Values.NAO)
            RHF.diis_space = 10
            # if (settings.chkpoint == True) and (settings.read_PHF == True) and (os.path.exists(f"{settings.mol_name}_RHFdm.p")):
            #     if settings.VERBOSE > 1:
            #         print(f"Loading RHF density from {settings.mol_name}_RHFdm.p")
            #     dm = pickle.load(open(f"{settings.mol_name}_RHFdm.p", "rb"))
            #     RHF.kernel(dm)
            # else:
            if n == 0:
                RHF.kernel()
            else:
                dm = np.dot(
                    guessRHF[:, : Values.NOccAO], guessRHF[:, : Values.NOccAO].T
                )
                RHF.kernel(dm)
            if settings.VERBOSE > 0:
                print("E(RHF)=", RHF.e_tot)
            "Checkpoint RHF"
            # if settings.chkpoint == True:
            #     pickle.dump(RHF.make_rdm1(), open(f"{settings.mol_name}_RHFdm.p", "wb"))
            A2R = np.zeros([Values.NSO, Values.NSO])
            A2R[: Values.NAO, : Values.NAO] = A2R[
                Values.NAO : Values.NSO, Values.NAO : Values.NSO
            ] = RHF.mo_coeff

            "check RHF stability"
            RHFstab = RHF.stability(external=True, return_status=True)
            if settings.is_RHF == False:
                break
            if RHFstab[2] == True:
                "Found internally stable RHF"
                break
            elif n == 9:
                raise Exception("Could not converge RHF")
            guessRHF = RHFstab[0]

        if settings.is_RHF:
            newA2R = SortOrb(
                A2R, Values.NAO, Values.NOccAO, Values.NSO, Values.NOccSO, 1
            )
            Values.A2G = newA2R.copy()
            # if settings.chkpoint == True:
            #     pickle.dump(Values.A2G, open(f"{settings.mol_name}_GHFMO.p", "wb"))

            return Values

        # print(RHF.converged)
        # raise


        ## UHF
        for n in range(10):
            "Run UHF until it is internally stable"
            UHF = scf.UHF(Values.mol)
            UHF.get_hcore = lambda *args: Values.h1
            UHF.get_ovlp = lambda *args: Values.Ovlp
            UHF._eri = pyscf.ao2mo.restore(8, Values.eri, Values.NAO)
            UHF.diis_space = 10
            UHF.level_shift = 0.1
            # if (settings.chkpoint == True) and (settings.read_PHF == True) and (os.path.exists(f"{settings.mol_name}_UHFdm.p")):
            #     if settings.VERBOSE > 1:
            #         print(f"Loading UHF density from {settings.mol_name}_UHFdm.p")
            #     dm = pickle.load(open(f"{settings.mol_name}_UHFdm.p", "rb"))
            #     UHF.kernel(dm)
            # else:
            if n == 0:
                guessUHF = RHFstab[1]
            else:
                guessUHF = UHFstab[0]
            dm1 = np.dot(
                guessUHF[0][:, : Values.NOccAO], guessUHF[0][:, : Values.NOccAO].T
            )
            dm2 = np.dot(
                guessUHF[1][:, : Values.NOccAO], guessUHF[1][:, : Values.NOccAO].T
            )
            dm = np.array([dm1, dm2])
            UHF.kernel(dm)
            "check UHF stability"
            UHFstab = UHF.stability(external=True, return_status=True)
            if UHFstab[2] == True:
                "Found internally stable UHF"
                break
            elif n == 9:
                raise Exception("Could not converge UHF")
        print(UHFstab[2], UHFstab[3])
        # Print results
        if settings.VERBOSE > 0:
            print("E(UHF)=", UHF.e_tot)
        # if settings.chkpoint == True:
        #     pickle.dump(UHF.make_rdm1(), open(f"{settings.mol_name}_UHFdm.p", "wb"))
        A2U = np.zeros([Values.NSO, Values.NSO])
        A2U[: Values.NAO, : Values.NAO] = UHF.mo_coeff[0]
        A2U[Values.NAO : Values.NSO, Values.NAO : Values.NSO] = UHF.mo_coeff[1]
        # R2U = A2R.T @ A2U
        # newA2R = SortOrb(A2R, Values.NAO, Values.NOccAO, Values.NSO, Values.NOccSO, 1)
        # newR2U = SortOrb(R2U, Values.NAO, Values.NOccAO, Values.NSO, Values.NOccSO, 1)
        # newA2U = SortOrb(A2U, Values.NAO, Values.NOccAO, Values.NSO, Values.NOccSO, 1)

        if settings.SP == 2:
            newA2U = SortOrb(
                A2U, Values.NAO, Values.NOccAO, Values.NSO, Values.NOccSO, 1
            )
            Values.A2G = newA2U.copy()
            # if settings.chkpoint == True:
            #     pickle.dump(Values.A2G, open(f"{settings.mol_name}_GHFMO.p", "wb"))

            return Values

        ## GHF
        GHF = scf.GHF(Values.mol)
        GHF.get_hcore = lambda *args: block_diag(Values.h1, Values.h1)
        GHF.get_ovlp = lambda *args: block_diag(Values.Ovlp, Values.Ovlp)
        GHF._eri = pyscf.ao2mo.restore(8, Values.eri, Values.NAO)
        GHF.diis_space = 10
        GHF.level_shift = 0.1
        if (
            (settings.chkpoint == True)
            and (settings.read_PHF == True)
            and (os.path.exists(f"{settings.mol_name}_GHFdm.p"))
        ):
            if settings.VERBOSE > 1:
                print(f"Loading GHF density from {settings.mol_name}_GHFdm.p")
            dm = pickle.load(open(f"{settings.mol_name}_GHFdm.p", "rb"))
            GHF.kernel(dm)
        else:
            guessGHF = UHFstab[1]
            guessGHF = SortOrb(
                guessGHF, Values.NAO, Values.NOccAO, Values.NSO, Values.NOccSO, 1
            )
            dm = np.dot(guessGHF[:, : Values.NOccSO], guessGHF[:, : Values.NOccSO].T)
            GHF.kernel(dm)
        GHFstab = GHF.stability(external=True)
        if settings.VERBOSE > 0:
            print("E(GHF)=", GHF.e_tot)
        # if settings.chkpoint == True:
        #     pickle.dump(GHF.make_rdm1(), open(f"{settings.mol_name}_GHFdm.p", "wb"))

        Values.A2G = GHF.mo_coeff.copy()
        # if settings.chkpoint == True:
        #     pickle.dump(Values.A2G, open(f"{settings.mol_name}_GHFMO.p", "wb"))
    else:
        C = pickle.load(open(f"{settings.orb_init}", "rb"))
        ovlp = Values.mol.intor("int1e_ovlp")
        n, o = Values.OAO.shape
        S = np.zeros([2 * n, 2 * o])
        S[:n, :o] = Values.OAO.T.dot(ovlp)
        S[n:, o:] = Values.OAO.T.dot(ovlp)
        Values.A2G = S.dot(C)
        "orb_init must be orthonormal (the kernel uses it as a unitary reference; a non-unitary one"
        "breaks [H, R] = 0 in the MO representation -> non-Hermitian projected H, wrong SGHF gradient)."
        "Check, Lowdin-orthonormalise a recoverable input (warn if the correction is appreciable),"
        "reject a rank-deficient one."
        A = Values.A2G; no = Values.NOccSO
        dev_in = float(np.abs(A.conj().T @ A - np.eye(A.shape[1])).max())
        w_occ = np.linalg.eigvalsh(A[:, :no].conj().T @ A[:, :no])
        if w_occ.min() < 1e-8 * max(w_occ.max(), 1e-300):
            raise ValueError(f"orb_init rejected: occupied orbitals are (nearly) linearly dependent "
                             f"(Gram eigenvalues {w_occ.min():.2e} .. {w_occ.max():.2e})")
        occ = A[:, :no]; w, v = np.linalg.eigh(occ.conj().T @ occ)
        occ = occ @ v @ np.diag(w ** -0.5) @ v.conj().T
        vir = A[:, no:] - occ @ (occ.conj().T @ A[:, no:])        # project out the occupied space
        w_vir = np.linalg.eigvalsh(vir.conj().T @ vir)
        if vir.shape[1] > 0 and w_vir.min() < 1e-8 * max(w_vir.max(), 1e-300):
            raise ValueError(f"orb_init rejected: virtual orbitals are (nearly) linearly dependent after "
                             f"projection on the occupied space (Gram eigenvalues {w_vir.min():.2e} .. {w_vir.max():.2e})")
        w, v = np.linalg.eigh(vir.conj().T @ vir)
        vir = vir @ v @ np.diag(w ** -0.5) @ v.conj().T
        Values.A2G = np.hstack((occ, vir))
        dev_out = float(np.abs(Values.A2G.conj().T @ Values.A2G - np.eye(A.shape[1])).max())
        if dev_in > 1e-6:
            print(f"WARNING: orb_init was not orthonormal (max|C^H C - 1| = {dev_in:.2e}); Lowdin-orthonormalised "
                  f"(now {dev_out:.1e}). The occupied span was kept; check that the seed is what you intended.")
        elif settings.VERBOSE > 1:
            print(f"orb_init: orthonormality deviation {dev_in:.1e} (ok)")

    return Values

    ## GCCSD
    # gcc = cc.UCCSD(UHF)
    # if (os.path.exists("t1t2.p")):
    # 	t1, t2 = pickle.load(open( "t1t2.p", "rb" ))
    # 	print("load GCCSD T")
    # 	gcc.kernel(t1,t2)
    # else:
    # 	gcc.kernel()
    # pickle.dump([gcc.t1,gcc.t2],open( "t1t2.p", "wb" ))
    # print('E(GCCSD)= %8.8f' %(gcc.e_tot + Enuc))
    # et = gcc.ccsd_t()
    # print('E(GCCSD-T)= %8.8f' %(et + gcc.e_tot + Enuc))
