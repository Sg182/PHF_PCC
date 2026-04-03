import numpy as np
import scipy.linalg as sl
from pyscf import gto, scf
import warnings
import pickle

np.set_printoptions(precision=14, threshold=np.inf, suppress=True, linewidth=200000000)

"""
This file contains the code used to construct initial guesses for PHF
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
    "NA":11,
    "MG":12,
    "AL":13,
    "SI":14,
    "P":15,
    "S":16,
    "CL":17,
    "AR":18,
}


def xyz_reader(file, VERBOSE=1):
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

    if len(lines) < n + 2:
        raise ValueError(
            "xyz file does not contain enough lines for the number of atoms!"
        )
    if (len(lines) > n + 2) and (VERBOSE > 0):
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
    return atom_num, coord


def make_mol(geom_file, basis, unit="Ang"):
    mol = gto.Mole()
    atom_num, coord = xyz_reader(geom_file)
    mol.atom = []
    for i in range(len(atom_num)):
        mol.atom.append([atom_num[i], coord[i, :]])
    mol.spin = 0
    mol.basis = basis
    mol.unit = unit
    mol.build()

    return mol


def SortOrb(MO, NAO, NOccAO, NSO, NOccSO, Type):
    # if type = 1
    # rearrange orbitals from OA VA  0  0 to OA  0 VA  0
    #                          0  0 OB VB     0 OB  0 VB
    # if type = 0, do the inverse
    if Type == 1:
        NVrtAO = NAO - NOccAO
        newMO = np.zeros([NSO, NSO])
        newMO[:, :NOccAO] = MO[:, :NOccAO]
        newMO[:, NOccAO:NOccSO] = MO[:, NAO : NAO + NOccAO]
        newMO[:, NOccSO : NOccSO + NVrtAO] = MO[:, NOccAO : NOccAO + NVrtAO]
        newMO[:, NAO + NOccAO : NSO] = MO[:, NAO + NOccAO : NSO]
        return newMO
    elif Type == 0:
        NVrtA = NAO - NOccAO
        newMO = np.zeros([NSO, NSO])
        newMO[:, :NOccAO] = MO[:, :NOccAO]
        newMO[:, NOccAO:NAO] = MO[:, NOccSO : NOccSO + NVrtA]
        newMO[:, NAO : NAO + NOccAO] = MO[:, NOccAO:NOccSO]
        newMO[:, NAO + NOccAO : NSO] = MO[:, NAO + NOccAO : NSO]
        return newMO


def init_guess_FMO(mol, frags, spins, parity, HF_type="RHF"):
    """
    Make init guess for current system
    """

    "Get fragment MOs"
    orbs = []
    for i, frag in enumerate(frags):
        new_mol = gto.Mole()
        new_mol.atom = []
        for x in frag:
            new_mol.atom.append(mol.atom[x])
        new_mol.spin = spins[i]
        new_mol.basis = mol.basis
        new_mol.unit = mol.unit
        new_mol.build()
        if HF_type == "RHF":
            HF = scf.RHF(new_mol)
            HF.kernel()
            C = HF.mo_coeff
            C = C[:, : new_mol.nelectron // 2]
        elif HF_type == "UHF":
            HF = scf.UHF(new_mol)
            HF.kernel()
            C_ = HF.mo_coeff
            C = []
            C.append(C_[0][:, : new_mol.nelec[0]])
            C.append(C_[1][:, : new_mol.nelec[1]])
        elif HF_type == "GHF":
            HF = scf.GHF(new_mol)
            HF.kernel()
            C = HF.mo_coeff
            C = C[:, : new_mol.nelectron]
        else:
            raise
        orbs.append(C)
    "Combine and optimize"
    if HF_type == "RHF":
        pass
    elif HF_type == "UHF":
        MO = [np.zeros((mol.nao, mol.nelec[0])), np.zeros((mol.nao, mol.nelec[1]))]
        x = 0
        y = 0
        z = 0
        for i, cur in enumerate(orbs):
            a, b = cur[0].shape
            _, c = cur[1].shape
            if parity[i] == 0:
                MO[0][x : x + a, y : y + b] = cur[0]
                MO[1][x : x + a, z : z + c] = cur[1]
                y += b
                z += c
            elif parity[i] == 1:
                MO[0][x : x + a, y : y + c] = cur[1]
                MO[1][x : x + a, z : z + b] = cur[0]
                y += c
                z += b
            x += a

        HF = scf.UHF(mol)
        dm1 = np.dot(MO[0], MO[0].T)
        dm2 = np.dot(MO[1], MO[1].T)
        dm = np.array([dm1, dm2])
        HF.kernel(dm)

        C = HF.mo_coeff

        if True:
            # UCCSD
            from pyscf import cc
            mfcc = cc.CCSD(HF)
            mfcc.direct= True
            mfcc.run()
            raise

        "Convert to GHF form"
        D = np.zeros([mol.nao * 2, mol.nao * 2])
        D[: mol.nao, : mol.nao] = C[0]
        D[mol.nao :, mol.nao :] = C[1]
        D = SortOrb(D, mol.nao, mol.nelec[0], 2 * mol.nao, mol.nelectron, 1)

    elif HF_type == "GHF":
        pass
    else:
        raise

    return D


def init_guess_VB(mol, HF_type="RHF"):
    """
    Make init guess for current system from the VB-PP configuration gathered
    from the VB2000 code
    """
    # Methane
    C = np.array(
        [
            [
                0.99683,
                0.99683,
                0.14442,
                0.08337,
                0.14442,
                0.08337,
                -0.14442,
                -0.08337,
                0.14442,
                0.08337,
            ],
            [
                0.02433,
                0.02433,
                -0.36778,
                -0.07704,
                -0.36778,
                -0.07704,
                0.36778,
                0.07704,
                -0.36778,
                -0.07704,
            ],
            [
                -0.01805,
                -0.01805,
                -0.30781,
                -0.12711,
                -0.30781,
                -0.12711,
                0.30781,
                0.12711,
                -0.30781,
                -0.12711,
            ],
            [
                0.00000,
                0.00000,
                -0.31108,
                -0.06983,
                -0.31108,
                -0.06983,
                -0.31108,
                -0.06983,
                0.31108,
                0.06983,
            ],
            [
                -0.00000,
                -0.00000,
                0.31108,
                0.06983,
                -0.31108,
                -0.06983,
                0.31108,
                0.06983,
                0.31108,
                0.06983,
            ],
            [
                0.00000,
                0.00000,
                0.31108,
                0.06983,
                -0.31108,
                -0.06983,
                -0.31108,
                -0.06983,
                -0.31108,
                -0.06983,
            ],
            [
                -0.00000,
                -0.00000,
                -0.12577,
                -0.05708,
                -0.12577,
                -0.05708,
                -0.12577,
                -0.05708,
                0.12577,
                0.05708,
            ],
            [
                0.00000,
                0.00000,
                0.12577,
                0.05708,
                -0.12577,
                -0.05708,
                0.12577,
                0.05708,
                0.12577,
                0.05708,
            ],
            [
                0.00000,
                0.00000,
                0.12577,
                0.05708,
                -0.12577,
                -0.05708,
                -0.12577,
                -0.05708,
                -0.12577,
                -0.05708,
            ],
            [
                -0.00060,
                -0.00060,
                -0.00425,
                -0.00396,
                -0.07580,
                -0.55399,
                0.00425,
                0.00396,
                -0.00425,
                -0.00396,
            ],
            [
                0.00304,
                0.00304,
                0.03997,
                0.04308,
                -0.07907,
                -0.33996,
                -0.03997,
                -0.04308,
                0.03997,
                0.04308,
            ],
            [
                -0.00060,
                -0.00060,
                -0.00425,
                -0.00396,
                -0.00425,
                -0.00396,
                0.00425,
                0.00396,
                -0.07580,
                -0.55399,
            ],
            [
                0.00304,
                0.00304,
                0.03997,
                0.04308,
                0.03997,
                0.04308,
                -0.03997,
                -0.04308,
                -0.07907,
                -0.33996,
            ],
            [
                -0.00060,
                -0.00060,
                -0.00425,
                -0.00396,
                -0.00425,
                -0.00396,
                0.07580,
                0.55399,
                -0.00425,
                -0.00396,
            ],
            [
                0.00304,
                0.00304,
                0.03997,
                0.04308,
                0.03997,
                0.04308,
                0.07907,
                0.33996,
                0.03997,
                0.04308,
            ],
            [
                -0.00060,
                -0.00060,
                -0.07580,
                -0.55399,
                -0.00425,
                -0.00396,
                0.00425,
                0.00396,
                -0.00425,
                -0.00396,
            ],
            [
                0.00304,
                0.00304,
                -0.07907,
                -0.33996,
                0.03997,
                0.04308,
                -0.03997,
                -0.04308,
                0.03997,
                0.04308,
            ],
        ]
    )

    Cu = C[:, ::2]
    Cd = C[:, 1::2]

    # H2
    C = np.array(
        [
            [0.07400, 0.55754],
            [0.07131, 0.43461],
            [0.55754, 0.07400],
            [0.43461, 0.07131],
        ]
    )

    Cu = C[:, ::2]
    Cd = C[:, 1::2]

    # up = [0, 2, 4, 6, 8]
    # down = [1, 3, 5, 7, 9]
    # Cu = C[:,up]
    # Cd = C[:,down]

    "Orthogonalize"
    S = mol.intor("int1e_ovlp")

    val, vec = sl.eigh(Cu.T.dot(S).dot(Cu))
    x = np.diag(1.0 / np.sqrt(val))
    Sinv = vec.dot(x).dot(vec.T)
    Cu = Cu.dot(Sinv)

    val, vec = sl.eigh(Cd.T.dot(S).dot(Cd))
    x = np.diag(1.0 / np.sqrt(val))
    Sinv = vec.dot(x).dot(vec.T)
    Cd = Cd.dot(Sinv)

    "Make virtuals"
    val, vec = sl.eig(Cu.dot(Cu.T).dot(S))
    vir = abs(val) < 0.1
    temp = vec[:, vir]
    val, vec = sl.eigh(temp.T.conj().dot(S).dot(temp))
    temp = np.real(temp.dot(vec).dot(np.diag(1.0 / np.sqrt(val))))
    Cu = np.hstack((Cu, temp))

    val, vec = sl.eig(Cd.dot(Cd.T).dot(S))
    vir = abs(val) < 0.1
    temp = vec[:, vir]
    val, vec = sl.eigh(temp.T.conj().dot(S).dot(temp))
    temp = np.real(temp.dot(vec))  # .dot(np.diag(1./np.sqrt(val))))
    Cd = np.hstack((Cd, temp))
    # Why this?
    for i in range(Cd.shape[1]):
        Cd[:, i] /= np.sqrt(Cd[:, i].T.dot(S).dot(Cd[:, i]))

    "Convert to GHF form"
    D = np.zeros([mol.nao * 2, mol.nao * 2])
    D[: mol.nao, : mol.nao] = Cu
    D[mol.nao :, mol.nao :] = Cd
    D = SortOrb(D, mol.nao, mol.nelec[0], 2 * mol.nao, mol.nelectron, 1)

    return D


def param(name):
    if name == "N2":
        save_file = "N2_6_31G_orb"
        geom_file = "N2.xyz"
        basis = "cc-pvdz"
        unit = "Ang"
        frags = [[0], [1]]
        spins = [3, 3]
        parity = [0, 1]
    elif name == "Methane":
        save_file = "Methane_6_31G_orb"
        geom_file = "Methane.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1], [2], [3], [4]]
        spins = [2, 1, 1, 1, 1]
        parity = [0, 0, 0, 1, 1]
    elif name == "H2":
        save_file = "H2_6_31G_orb"
        geom_file = "H2.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1]]
        spins = [1, 1]
        parity = [0, 1]
    elif name == "H2_D2h":
        save_file = "H2_D2h_6_31G_orb"
        geom_file = "H2_D2h.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1], [2], [3]]
        spins = [1, 1, 1, 1]
        parity = [0, 1, 1, 0]
    elif name == "F2":
        save_file = "F2_6_31G_orb"
        geom_file = "F2.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1]]
        spins = [1, 1]
        parity = [0, 1]
    elif name == "C2":
        save_file = "C2_6_31G_orb"
        geom_file = "C2.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1]]
        spins = [2, 2]
        parity = [0, 1]
    elif name == "Be2":
        save_file = "Be2_6_31G_orb"
        geom_file = "Be2.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1]]
        spins = [0, 0]
        parity = [0, 1]
    elif name == "CO":
        save_file = "CO_6_31G_orb"
        geom_file = "CO.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1]]
        spins = [2, 2]
        parity = [0, 1]
    elif name == "NaCl":
        save_file = "NaCl_6_31G_orb"
        geom_file = "NaCl.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1]]
        spins = [1, 1]
        parity = [0, 1]
    elif name == "LiF":
        save_file = "LiF_6_31G_orb"
        geom_file = "LiF.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1]]
        spins = [1, 1]
        parity = [0, 1]
    elif name == "H6_ring":
        save_file = "H6_ring_6_31G_orb"
        geom_file = "H6_ring.xyz"
        basis = "6-31G"
        unit = "Ang"
        frags = [[0], [1], [2], [3], [4], [5]]
        spins = [1, 1, 1, 1, 1, 1]
        parity = [0, 1, 0, 1, 0, 1]
    else:
        raise("Bad name")
    return save_file, geom_file, basis, unit, frags, spins, parity


def driver():
    name = "N2"
    save_file, geom_file, basis, unit, frags, spins, parity = param(name)
    mol = make_mol(geom_file, basis, unit=unit)
    # D = init_guess_VB(mol, HF_type="UHF")
    D = init_guess_FMO(mol, frags, spins, parity, HF_type="UHF")
    pickle.dump(D, open(save_file, "wb"))


if __name__ == "__main__":
    driver()
