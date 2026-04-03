#!/bin/sh

# These compile commands assume that f2py is using meson to build

# mpi version
# export FC="mpif90"
# export F90="mpif90"
# export CC="mpicc"
# export FFLAGS="-fopenmp -fPIC -O3"
# export LDFLAGS="-lmpi -lmpi_mpifh"
# export LDSHARED="mpif90"
# f2py -lgomp -m PUCC_X2 -c Precision.f90 Constants.f90 Wrappers.f90 UHF.f90 IntTrans.f90 CCRes.f90 PCCCI.f90 CCSD_T.f90 FPUCC_Tools.f90 FPUCC.f90 Broyden.f90 Spin.f90 Main.f90 -I/home/jk220/Documents/PHF_PCC/SUCCSD_Boson_cmplx/ 
 

# No mpi version
f2py --f90exec=gfortran --f77exec=gfortran --opt='-O3' --f90flags="-fopenmp" -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lgomp -m PUCC_X2_nompi -c Precision.f90 Constants.f90 Wrappers.f90 UHF.f90 IntTrans.f90 CCRes.f90 PCCCI.f90 CCSD_T.f90 FPUCC_Tools.f90 FPUCC_nompi.f90 Update_nompi.f90 DIIS.f90 Broyden_nompi.f90 Spin_nompi.f90 Main_nompi.f90 -I/home/jk220/Documents/PHF_PCC/SUCCSD_Boson_cmplx/ 

# export FC=""
# export CC="gcc"
# export FFLAGS="-fopenmp -fPIC -O3"
# export LDFLAGS="-lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lgomp"
# f2py -m PUCC_X2_nompi -c Precision.f90 Constants.f90 Wrappers.f90 UHF.f90 IntTrans.f90 CCRes.f90 PCCCI.f90 CCSD_T.f90 FPUCC_Tools.f90 FPUCC_nompi.f90 Update.f90 DIIS.f90 Broyden_nompi.f90 Spin_nompi.f90 Main_nompi.f90 -I/home/jk220/Documents/PHF_PCC/SUCCSD_Boson_cmplx/ 

#
# gfortran -fopenmp -O3 -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lgomp -c Precision.f90 Constants.f90 Wrappers.f90 UHF.f90 IntTrans.f90 CCRes.f90 PCCCI.f90 CCSD_T.f90 FPUCC_Tools.f90 FPUCC_nompi.f90 Update.f90 DIIS.f90 Broyden_nompi.f90 Spin_nompi.f90 Main_nompi.f90 -I/home/jk220/Documents/PHF_PCC/SUCCSD_Boson_cmplx/ 

#rm *mod
#rm *.o
