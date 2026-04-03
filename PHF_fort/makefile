#!/bin/sh

# These compile commands assume that f2py is using meson to build

# mpi version
# export FC="mpif90"
# export F90="mpif90"
# export CC="mpicc"
# export FFLAGS="-fopenmp -fPIC -O3"
# export LDFLAGS="-lmpi -lmpi_mpifh"
# export LDSHARED="mpif90"
# f2py -m PHFTools_F -c Precision.f90 Constants.f90 BuildSC_dr.f90 Wrappers.f90 PHFTools.f90 -L/home/jk220/Documents/PHF_PCC/PHF_fort -lAllLinAlg -lgomp 

# no mpi version
# export FC="gfortran"
# export CC="gcc"
# export FFLAGS="-fopenmp -fPIC -O3"
f2py -lgomp -m PHFTools_nompi_F -c Precision.f90 Constants.f90 BuildSC_dr.f90 Wrappers.f90 PHFTools_nompi.f90 -L. -L/home/jk220/Documents/PHF_PCC/PHF_fort -lAllLinAlg

#rm *mod
#rm *.o
