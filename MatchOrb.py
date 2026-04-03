import numpy as np
import scipy.linalg as sl

def MatchOrb(MOs,Ref,nocc):
    """
    Rotate MOs to look more like Ref
    Both are assumed in an orthogonal basis
    """
    occ0 = Ref[:,:nocc]
    occ1 = MOs[:,:nocc]
    vir0 = Ref[:,nocc:]
    vir1 = MOs[:,nocc:]
    m = occ0.T.conj() @ occ1
    u,s,vh = np.linalg.svd(m)
    v = vh.conj().T
    uh = u.conj().T
    print("occ ovlp=",s)
    occ = occ1 @ v @ uh
    m = vir0.T.conj() @ vir1
    a,b = m.shape
    if a != b:
        val, vec = sl.eigh(-m.T.conj().dot(m))
        print("vir ovlp=",np.sqrt(np.abs(val)))
        c = np.where(np.abs(val) > 0.1)[0]
        d = np.zeros(len(val),dtype=bool)
        d[c] = True
        temp = vir1.dot(vec[:,d])
        temp2 = vir1.dot(vec[:,~d])
        m = vir0.T.conj() @ temp
        u,s,vh = np.linalg.svd(m)
        v = vh.conj().T
        uh = u.conj().T
        print("vir ovlp=",s)
        vir = np.hstack((temp @ v @ uh,temp2))
    else:
        u,s,vh = np.linalg.svd(m)
        v = vh.conj().T
        uh = u.conj().T
        print("vir ovlp=",s)
        vir = vir1 @ v @ uh
    newMO = 0 * MOs
    newMO[:,:nocc] = occ
    newMO[:,nocc:] = vir
    return newMO
