import numpy as np
import scipy.linalg as sl


def ao2mo(mat, MOs, dim):
    "transfer integral from ao to mo"
    if dim == 2:
        newmat = np.einsum("ai,ab,bj -> ij", MOs.conj(), mat, MOs, optimize="optimal")
    if dim == 4:
        newmat = np.einsum(
            "ai,bj,abcd,ck,dl -> ijkl",
            MOs.conj(),
            MOs.conj(),
            mat,
            MOs,
            MOs,
            optimize="optimal",
        )
    return newmat


def getTrans(Ovlp):
    tol = 1e-10
    evals, evecs = sl.eigh(Ovlp, lower=False)
    invS = [np.sqrt(abs(x)) / (x + tol) for x in evals]
    Sinv = np.diag(invS)
    X = evecs.dot(Sinv)
    return X
