import numpy as np
from SpinSymm import *


def Spin_Proj(SP, ngrid, NAO, NSO, J):
    """
    Prepare roots and weights for spin projection
    ################################################################################################
    ==Inputs==
    SP : int
     Determines type of spin projection
    ngrid : list
     Number of grid points for both Lebedev grids
    NAO : int
     Number of AOs
    NSO : int
     Number of spin orbitals
    J : int
     J values for desired spin
    ===============================================================================================
    ==Outputs==
    nci : int
     Number of ci elements needed to determine after projection
    roota : float array
     Grid points in the alpha rotation
    rootb : float array
     Grid points in the beta rotation
    rooty : float array
     Grid points in the gamma rotation
    weights : float array
     integration weights for each grid point
    R1 : complex array
      Spin orbital rotation matrix for each of the first grid points
    R1 : complex array
      Spin orbital rotation matrix for each of the second grid points
    """
    if SP == 0:
        "No spin projection, make R"
        nci = 1
        weights = np.ones([1, 1])
        R1 = R2 = np.zeros([1, NSO, NSO], dtype=complex)
        R1[0, :, :] = np.eye(NSO)
        R2[0, :, :] = np.eye(NSO)
        roota = rootb = rooty = np.zeros([1])
    if SP == 1:
        nci = 2 * J + 1
        roota, rootb, rooty, weights = RootWeight(ngrid)
        R1 = np.zeros([ngrid[0], NSO, NSO], dtype=complex)
        R2 = np.zeros([ngrid[1], NSO, NSO], dtype=complex)
        for i in range(ngrid[0]):
            R1[i, :, :] = BuildRotMat(roota[i], rootb[i], 0, NAO, NSO)
        for i in range(ngrid[1]):
            R2[i, :, :] = BuildRotMat(0, 0, rooty[i], NAO, NSO)
    if SP == 2:
        nci = 1
        roota = rooty = np.zeros([1])
        rootb, weightb = RootWeightSy(ngrid[1])
        weights = np.zeros([1, ngrid[1]])
        weights[0, :] = weightb
        R1 = np.zeros([1, NSO, NSO], dtype=complex)
        R2 = np.zeros([ngrid[1], NSO, NSO], dtype=complex)
        R1[0, :, :] = np.eye(NSO)
        for i in range(ngrid[1]):
            R2[i, :, :] = BuildRotMat(0, rootb[i], 0, NAO, NSO)
    if SP == 3:
        nci = 1
        roota, rootb, rooty, weights = RootWeight(ngrid)
        R1 = np.zeros([ngrid[0], NSO, NSO], dtype=complex)
        R2 = np.zeros([ngrid[1], NSO, NSO], dtype=complex)
        R1[0, :, :] = np.eye(NSO)
        for i in range(ngrid[1]):
            R2[i, :, :] = BuildRotMat(0, 0, rooty[i], NAO, NSO)
    return nci, roota, rootb, rooty, weights, R1, R2
