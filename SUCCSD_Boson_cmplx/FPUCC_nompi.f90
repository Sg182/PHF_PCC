Module FPUCC_nompi
  Use Precision
  Use Constants
  Use IntTrans
  Use UHF
  Use CCSD_T
  Use FPUCC_Tools 
  Implicit None
  Logical  :: TrunT1 = .true.

  Contains

    Subroutine PbarHbarOlap(Olap0,OlapEx1,OlapEx2,EOlap0,EOlapEx1, &
                            EOlapEx2,HOne,HTwo,Ref,T1,T2,NOcc,NAO,NSO,ENuc,QJ, SP, &
                            NPoints, Npg, ncisp, ncipg, ncik, R1, R2, Rpg, Rk, &
                            Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk,TrunODE)
    Implicit None
    !------------------------------------------------------------------------------------------------
    ! Calculate components for PCCSD residuals, Eq 11
    !################################################################################################
    ! ==Inputs==
    ! NOcc : int
    !   Number of occupied spin orbitals
    ! NSO : int
    !   Number of spin oribtals
    ! NAO : int
    !   Number of atomic oribtals
    ! Enuc : real
    !   Energy of nuclear repulsion and core if any
    ! NPoints : int
    !   Number of grid points used for spin projection(2 grids)
    ! Npg : int
    !   Number of grid points in point group projection(Number of group elements)
    ! QJ : int
    !   J value being projected into for spin projection
    ! SP : int
    !   Parameter controlling if spin projection is done
    ! nci : int
    !   Total number of CI elements from all projections
    ! ncik : int
    !   Parameter controlling if complex conjugation projection is done
    ! ncisp : int
    !   Number of CI elements coming from spin projection
    ! ncipg : int
    !   Number of CI elements coming from point group projection
    ! HOne : complex
    !   One electron integrals
    ! HTwo : complex
    !   Two electron integrals
    ! Ref : complex
    !   Reference MOs
    ! T1 : complex
    !   Current singles amplitudes
    ! T2 : complex
    !   Current doubles amplitudes
    ! R1 : complex
    !   Spin orbital rotation matrix for each of the first grid points
    ! R2 : complex
    !   Spin orbital rotation matrix for each of the second grid points
    ! Rpg : complex
    !   Spin orbital rotation matrix for each of the point group elements
    ! Weightsp : real
    !   Integration weights for each spin projection grid point
    ! Weightpg : real
    !   Integration weights for each point group projection grid point
    ! Roota : real
    !   Grid points in the alpha rotation for spin projection
    ! Rootb : real
    !   Grid points in the beta rotation for spin projection
    ! Rooty : real
    !   Grid points in the gamma rotation for spin projection
    ! fsp : complex
    !   Coefficients from PHF diagonalization for spin projection
    ! fpg : complex
    !   Coefficients from PHF diagonalization for point group projection
    ! fk : complex
    !   Coefficients from PHF diagonalization for complex conjugation projection
    ! TrunODE : int
    !   Value controlling excitation degree for U and W
    ! ===============================================================================================
    ! ==Outputs==
    ! Olap0 : complex
    !   N?
    ! OlapEx1 : complex
    !   NN_mu singles?
    ! OlapEx2 : complex
    !   NN_mu doubles?
    ! EOlap0 : complex
    !   NHbar?
    ! EOlapEx1 : complex
    !   NHbar_mu singles?
    ! EOlapEx2 : complex
    !   NHbar_mu doubles?
    !------------------------------------------------------------------------------------------------
    Integer,           Intent(In)  :: NOcc, NAO, NSO
    Integer,           Intent(In)  :: NPoints(2), Npg
    Integer,           Intent(In)  :: TrunODE, QJ, SP
    Integer,           Intent(In)  :: ncisp, ncipg, ncik
    Real    (Kind=pr), Intent(In)  :: Roota(:), Rootb(:), Rooty(:)
    Real    (Kind=pr), Intent(In)  :: Weightsp(:,:), Weightpg(Npg,ncipg,ncipg)
    Complex (Kind=pr), Intent(In)  :: fsp(ncisp),fpg(ncipg),fk(ncik)
    Complex (Kind=pr), Intent(In)  :: R1(:,:,:), R2(:,:,:)
    Complex (Kind=pr), Intent(In)  :: Rpg(npg,NSO,NSO), Rk(ncik,NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: T1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr), Intent(In)  :: T2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    Complex (Kind=pr), Intent(In)  :: HOne(NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: HTwo(NSO,NSO,NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: Ref(NSO,NSO)
    Real (Kind=pr),    Intent(In)  :: ENuc 
    Complex (Kind=pr), Intent(Out) :: Olap0, EOlap0 
    Complex (Kind=pr), Intent(Out) :: OlapEx1(NOcc+1:NSO,NOcc) 
    Complex (Kind=pr), Intent(Out) :: OlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc) 
    Complex (Kind=pr), Intent(Out) :: EOlapEx1(NOcc+1:NSO,NOcc) 
    Complex (Kind=pr), Intent(Out) :: EOlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc) 
    ! Transformed Hamiltonian
    Complex (Kind=pr), Allocatable :: Fock(:,:)
    Complex (Kind=pr), Allocatable :: T1Loc(:,:) 
    ! Thouless
    Complex (Kind=pr), Allocatable :: RotMat(:,:), V1Del(:,:), V1Old(:,:)
    Complex (Kind=pr)              :: OvlTh
    ! Utildes
    Complex (Kind=pr), Allocatable :: U1t(:,:), U2t(:,:,:,:)
    Complex (Kind=pr), Allocatable :: U3t(:,:,:,:,:,:)
    Complex (Kind=pr), Allocatable :: U4t(:)
    Complex (Kind=pr)              :: STot 
    Complex (Kind=pr)              :: U0t, STotChk 
    Integer                        :: O, V, nq 
    Real (Kind=pr)                 :: MaxImag
    ! Spin integration stuff
    Integer                        :: I, J, K, M, N, P, Q
    Integer                        :: IPointA, IPointB, IPointY, IPG, IK
    Integer                        :: IAlloc
    Real (Kind=pr)                 :: Beta, Beta0
    Real (Kind=pr)                 :: BetaH
    Real (Kind=pr)                 :: Alpha, Gama
    Complex (Kind=pr)              :: Euler(3), wig 
    Complex (Kind=pr), Allocatable :: iSyBK(:,:), iSzBK(:,:)
    ! For excited Kernels
    Complex (Kind=pr)              :: ESCF 
    Complex (Kind=pr)              :: eps0, lambda0
    Complex (Kind=pr)              :: WeightTot 
    ! Complex (Kind=pr), Allocatable :: tmp1(:,:), tmp2(:,:,:,:)
    Complex (Kind=pr), Allocatable :: eps1(:,:), eps2(:,:,:,:)
    Complex (Kind=pr), Allocatable :: lambda1(:,:), lambda2(:,:,:,:)
    ! RK variables
    Real (Kind=pr)                 :: hin, TolRK
    Logical                        :: ODESuccess
    ! Restart stuff
    Logical                        :: Exists
    ! Timers
    Integer         :: cnt1, cnt2, clock_rate, clock_max
    Real (Kind=pr)  :: cput1, cput2
    Logical         :: ShowTime = .True.
    ! openmp
    INTEGER, EXTERNAL :: OMP_GET_THREAD_NUM, OMP_GET_NUM_THREADS

    !====================================================================!
    ! This is the driver for SGCCSD Residual                             !
    !====================================================================!
    ! Write(*,*) "Entered PbarHbarRes"
    Allocate(RotMat(NSO,NSO), T1Loc(NOcc+1:NSO,NOcc),         &
             V1Del(NOcc,NOcc+1:NSO), V1Old(NOcc,NOcc+1:NSO),  &
             Fock(NSO,NSO),     &
             ! tmp1(NSO,NSO), tmp2(NSO,NSO,NSO,NSO),                        &
             eps1(NSO-NOcc,NOcc), eps2(NSO-NOcc,NSO-NOcc,NOcc,NOcc),            &
             lambda1(NSO-NOcc,NOcc), lambda2(NSO-NOcc,NSO-NOcc,NOcc,NOcc),      &
             Stat=IAlloc)
    If(IAlloc /= 0) Stop "Could not allocate in PbarHbarRes"
    ! The coding of TrunODE is 
    ! 1: SD; SD
    ! 2: SDt; SD
    ! 3: SDt; SDtq
    ! 4: SDT; SDT
    ! 5: SDTq; SDT
    ! 6: SDTq; SDTq
    ! 7: SDTQ; SDTQ

    ! Write(*,*) "max|T1|", maxval(abs(T1))
    ! Write(*,*) "max|T2|", maxval(abs(T2))
    ! Write(*,*) "max|imag(T1)|", maxval(abs(aimag(T1)))
    ! Write(*,*) "max|imag(T2)|", maxval(abs(aimag(T2)))
    ! Initialize the overlap matrices 
    EOlap0= Zero
    EOlapEx1= Zero
    EOlapEx2= Zero
    Olap0 = Zero
    OlapEx1 = Zero
    OlapEx2 = Zero
    V1Old = Zero

    ! Initial condition for Utildes
    STot= One 
    Beta0 = Zero
    ! Write(6,50)
    ! Write(6,51)
    ! Write(6,50)
    ! 50  Format('**************************************************')
    ! 51  Format('*                 Start integration              *')

    ! print *, '**************************************************'
    ! print *, '*                 Start integration              *'
    ! print *, '**************************************************'
    betaH = 0.00_pr
    Euler = Zero
    Open(2048,File='Kern.log')
    ! 2049  Format(I5,1x,F12.8,1x,F12.8,1x,F12.8)
    ! Format(I5,1x,F12.8,1x,F12.8,1x,F12.8)
    !!$omp parallel default(shared)

    !!$omp do schedule(static) collapse(3), &
    !!$omp& reduction(+:EOlap0,Olap0,EOlapEx1,OlapEx1,EOlapEx2,OlapEx2), &
    !!$omp& private(RotMat,cput1,cnt1,cput2,cnt2,M,N,P,Q,wig,WeightTot), &
    !!$omp& private(eps0,lambda0,eps1,lambda1,eps2,lambda2), &
    !!$omp& private(IPointA,IPointY,IPG)
    Do IPointA = 1, NPoints(1)
    Do IPointY = 1, NPoints(2)
    Do IPG = 1, NPG
      ! Write(*,*) 'Grid info', IPointA, IPointY, IPG

      ! Build total projection matrix in SO basis
      RotMat = MatMul(R1(IPointA,:,:), R2(IPointY,:,:))
      RotMat = MatMul(RotMat, Rpg(IPG,:,:))

      Call cpu_time(cput1)
      Call system_clock(cnt1,clock_rate,clock_max)

      !Solve for residuals at each grid point
      Call EvalKernel(HOne, HTwo, T1, T2, ENuc, &
                      eps0, lambda0, eps1, eps2, lambda1, lambda2, &
                      RotMat, NOcc, NAO, NSO, &
                      SP,TrunODE)

      Call cpu_time(cput2)
      Call system_clock(cnt2,clock_rate,clock_max)

      ! If(Showtime) Write(*,*) 'Grid real time = ', real(cnt2-cnt1)/real(clock_rate)

      Do M = -QJ, QJ
      Do N = -QJ, QJ
      Do P = 1, ncipg
      Do Q = 1, ncipg
        if (SP == 2) then
          Call Wigner(QJ,M,N,Roota(IPointA),Rootb(IPointY),Rooty(IPointA),wig)
        else
          Call Wigner(QJ,M,N,Roota(IPointA),Rootb(IPointA),Rooty(IPointY),wig)
        end if
        WeightTot = Weightsp(IPointA,IPointY) * Weightpg(IPG,P,Q)
        WeightTot = wig * WeightTot * conjg(fsp(QJ+1+M)) * fsp(QJ+1+N)
        WeightTot = WeightTot * conjg(fpg(P)) * fpg(Q)
        EOlap0= EOlap0+ WeightTot * eps0
        Olap0 = Olap0 + WeightTot * lambda0
        EOlapEx1= EOlapEx1+ WeightTot * eps1
        OlapEx1 = OlapEx1 + WeightTot * lambda1
        EOlapEx2= EOlapEx2+ WeightTot * eps2
        OlapEx2 = OlapEx2 + WeightTot * lambda2
      End Do
      End Do
      End Do
      End Do
      flush(6)
    EndDo
    EndDo
    EndDo
    !!$omp end do


    If (ncik == 2) then
    !!$omp do schedule(static) collapse(3), &
    !!$omp& reduction(+:EOlap0,Olap0,EOlapEx1,OlapEx1,EOlapEx2,OlapEx2), &
    !!$omp& private(RotMat,cput1,cnt1,cput2,cnt2,M,N,P,Q,wig,WeightTot), &
    !!$omp& private(eps0,lambda0,eps1,lambda1,eps2,lambda2), &
    !!$omp& private(IPointA,IPointY,IPG)
    Do IPointA = 1, NPoints(1)
    Do IPointY = 1, NPoints(2)
    Do IPG = 1, NPG
      RotMat = MatMul(R1(IPointA,:,:), R2(IPointY,:,:))
      RotMat = MatMul(RotMat, Rpg(IPG,:,:))
      RotMat = MatMul(RotMat, Rk(2,:,:))
      Call EvalKernel(Conjg(HOne), Conjg(HTwo), Conjg(T1), Conjg(T2), ENuc, &
                      eps0, lambda0, eps1, eps2, lambda1, lambda2, &
                      RotMat, NOcc, NAO, NSO, &
                      SP,TrunODE)
      Do M = -QJ, QJ
      Do N = -QJ, QJ
      Do P = 1, ncipg
      Do Q = 1, ncipg
        if (SP == 2) then
          Call Wigner(QJ,M,N,Roota(IPointA),Rootb(IPointY),Rooty(IPointA),wig)
        else
          Call Wigner(QJ,M,N,Roota(IPointA),Rootb(IPointA),Rooty(IPointY),wig)
        end if
        WeightTot = Weightsp(IPointA,IPointY) * Weightpg(IPG,P,Q)
        WeightTot = wig * WeightTot * conjg(fsp(QJ+1+M)) * fsp(QJ+1+N)
        WeightTot = WeightTot * conjg(fpg(P)) * fpg(Q)
        WeightTot = WeightTot * fk(2)
        EOlap0= EOlap0+ WeightTot * eps0
        Olap0 = Olap0 + WeightTot * lambda0
        EOlapEx1= EOlapEx1+ WeightTot * eps1
        OlapEx1 = OlapEx1 + WeightTot * lambda1
        EOlapEx2= EOlapEx2+ WeightTot * eps2
        OlapEx2 = OlapEx2 + WeightTot * lambda2
      End Do
      End Do
      End Do
      End Do
    End Do
    End Do
    End Do
    !!$omp end do
    End If
    !!$omp end parallel
    Close(2048)
    ! 70    Format('Finished grid points(beta):',1(I5))
    ! 71    Format('Required grid points(beta):',1(I5))
    ! print *, 'Finished grid points(beta):',1(I5)
    ! print *, 'Required grid points(beta):',1(I5)
    if (SP == 2) then
      Call ProjSz(OlapEx1,OlapEx2,NOcc,NAO,NSO)
      Call ProjSz(EOlapEx1,EOlapEx2,NOcc,NAO,NSO)
    end if
    ! test real   
    !     iEOlap0 = cmplx(real(iEOlap0, kind=pr), zero, kind=pr)
    !     iOlap0 = cmplx(real(iOlap0, kind=pr), zero, kind=pr)
    !     iEOlapEx1 = cmplx(real(iEOlapEx1, kind=pr), zero, kind=pr)
    !     iOlapEx1 = cmplx(real(iOlapEx1, kind=pr), zero, kind=pr)
    !     iEOlapEx2 = cmplx(real(iEOlapEx2, kind=pr), zero, kind=pr)
    !     iOlapEx2 = cmplx(real(iOlapEx2, kind=pr), zero, kind=pr)
    
    ! Restore iSy, iSz
    ! Deallocate
    DeAllocate(Rotmat, T1Loc, V1Del, V1Old, Fock,    &
               eps1, eps2, lambda1, lambda2,  &
    !                HOneT1, HTwoT1,      &
    !                EOlap1, EOlap2, Olap1, Olap2, &
    !                tmp1, tmp2, &
               Stat=IAlloc)
    If(IAlloc /= 0) Stop "Could not deallocate in PbarHbarRes"
    !      Call ShutDownGridY
    !      Call ShutDownGridZ
    Return
    End Subroutine PbarHbarOlap

    Subroutine BuildSC_test(t1,t2,c1,c2,NOcc,NSO,Ngrid,Npg,QJ,SP, &
                       ncisp,ncipg,ncik,R1, R2, Rpg, Rk, &
                       Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk)
    Implicit None
    Integer,            Intent(in)  :: NOcc,NSO,Ngrid(2),Npg,QJ,SP
    Integer,            Intent(in)  :: ncisp,ncipg,ncik
    Real    (Kind=pr),  Intent(In)  :: Weightsp(:,:),Weightpg(Npg,ncipg,ncipg)
    Real    (Kind=pr),  Intent(In)  :: Roota(:), Rootb(:), Rooty(:)
    Complex (kind=pr),  Intent(in)  :: R1(:,:,:), R2(:,:,:)
    Complex (Kind=pr),  Intent(in)  :: Rpg(npg,NSO,NSO), Rk(ncik,NSO,NSO)
    Complex (kind=pr),  Intent(in)  :: fsp(ncisp),fpg(ncipg),fk(ncik)
    Complex (Kind=pr),  Intent(in)  :: c1(NSO,NSO)
    Complex (Kind=pr),  Intent(in)  :: c2(NSO,NSO,NSO,NSO)
    Complex (Kind=pr),  Intent(out) :: t1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr),  Intent(out) :: t2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    Complex (kind=pr),  allocatable :: R(:,:), a1(:,:)
    Complex (kind=pr),  allocatable :: a2(:,:,:,:)
    Integer :: I, J, ni, nj, p ,q, m, n, d, il, iy, ipg, ik

    Complex(kind=pr) :: s, ovlp
    Complex(kind=pr) :: w, wig
    ! openmp
    INTEGER, EXTERNAL :: OMP_GET_THREAD_NUM, OMP_GET_NUM_THREADS

    Allocate(a1(NSO,NSO),a2(NSO,NSO,NSO,NSO),R(NSO,NSO))
    t1 = Zero
    t2 = Zero
    ovlp = Zero
    !!$omp parallel default(shared)

    !!$omp do schedule(static) collapse(3), &
    !!$omp& reduction(+:ovlp,t1,t2), &
    !!$omp& private(R,s,a1,a2,M,N,P,Q,wig,w,il,iy,ipg)
    Do il = 1, Ngrid(1)
    Do iy = 1, Ngrid(2)
    Do ipg = 1, Npg
      R = MatMul(R1(il,:,:), R2(iy,:,:))
      R = MatMul(R, Rpg(ipg,:,:))
      Call EvalOvlp(C1,C2,s,a1,a2,R, NOcc,NSO, SP)
      s = weightsp(il,iy) * s
      a1 = a1 * s
      a2 = a2 * s
      Do m = -QJ, QJ
      Do n = -QJ, QJ
          if (SP == 2) then
              Call Wigner(QJ,M,N,Roota(il),Rootb(iy),Rooty(il),wig)
          else
              Call Wigner(QJ,M,N,Roota(il),Rootb(il),Rooty(iy),wig)
          end if
          Do p = 1, ncipg
          Do q = 1, ncipg
              w = weightpg(ipg,p,q) * wig
              w = w * conjg(fsp(m+1+QJ)) * fsp(n+1+QJ)
              w = w * conjg(fpg(p)) * fpg(q)
              t1 = t1 + a1(NOcc+1:NSO,:NOcc) * w
              t2 = t2 + a2(NOcc+1:NSO,NOcc+1:NSO,:NOcc,:NOcc) * w
              ovlp = ovlp + s * w
          End Do
          End Do
      End Do
      End Do
    End Do
    End Do
    End Do
    !!$omp end do
    !!$omp end parallel
    Deallocate(a1,a2,R)
    t1 = t1 / ovlp
    t2 = t2 / ovlp
    End Subroutine BuildSC_test

    Subroutine PbarHbarResFromOlap2(Res0,Res1,Res2,  &
               Ref,Ene,Olap0,OlapEx1,OlapEx2,EOlap0, &
               EOlapEx1,EOlapEx2,NOcc,NAO,NSO,SP)
    Integer,           Intent(In)  :: NOcc, NAO, NSO,SP
    Complex (Kind=pr), Intent(In)  :: Ref(NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: Olap0, OlapEx1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr), Intent(In)  :: OlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    Complex (Kind=pr), Intent(In)  :: EOlap0, EOlapEx1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr), Intent(In)  :: EOlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    Complex (Kind=pr), Intent(In)  :: Ene 
    Complex (Kind=pr), Intent(Out) :: Res0 
    Complex (Kind=pr), Intent(Out) :: Res1(NOcc+1:NSO,NOcc) 
    Complex (Kind=pr), Intent(Out) :: Res2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc) 
    Integer   :: I, J, A, B
    ! Build residues. 
    Res0 = EOlap0 - Ene*Olap0
    Do I = 1, NOcc
    Do A = NOcc+1, NSO
      Res1(A,I) = (EOlapEx1(A,I) - Ene*OlapEx1(A,I)) / Olap0
    Do J = 1, NOcc
    Do B = NOcc+1, NSO
      Res2(A,B,I,J) = (EOlapEx2(A,B,I,J) - Ene*OlapEx2(A,B,I,J) )/Olap0
    EndDo
    EndDo
    EndDo
    EndDo
    if (SP == 2) then
        Call ProjSz(Res1,Res2,NOcc,NAO,NSO)
    end if
    Return
    End Subroutine PbarHbarResFromOlap2

    Subroutine PbarHbarResFromOlap(Ene,Res0,Res1,Res2,  &
               Ref,Olap0,OlapEx1,OlapEx2,EOlap0,&
               EOlapEx1,EOlapEx2,NOcc,NAO,NSO,SP)
    Integer,           Intent(In)  :: NOcc, NAO, NSO,SP
    Complex (Kind=pr), Intent(In)  :: Ref(NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: Olap0, OlapEx1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr), Intent(In)  :: OlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    Complex (Kind=pr), Intent(In)  :: EOlap0, EOlapEx1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr), Intent(In)  :: EOlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    Complex (Kind=pr), Intent(Out) :: Ene, Res0 
    Complex (Kind=pr), Intent(Out) :: Res1(NOcc+1:NSO,NOcc) 
    Complex (Kind=pr), Intent(Out) :: Res2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc) 
    Integer   :: I, J, A, B
    ! Build residues. 
    Ene = EOlap0/Olap0
    Res0 = Zero
    Do I = 1, NOcc
    Do A = NOcc+1, NSO
      Res1(A,I) = (EOlapEx1(A,I) - Ene*OlapEx1(A,I)) / Olap0
    Do J = 1, NOcc
    Do B = NOcc+1, NSO
      Res2(A,B,I,J) = (EOlapEx2(A,B,I,J) - Ene*OlapEx2(A,B,I,J) )/Olap0
    EndDo
    EndDo
    EndDo
    EndDo
    if (SP == 2) then
        Call ProjSz(Res1,Res2,NOcc,NAO,NSO)
    endif
    Print *, "Ovlp, Ene =", Olap0, Ene
    ! Print *, "Max Res1 =", maxval(abs(Res1))
    ! Print *, "Max Res2 =", maxval(abs(Res2))
    Return
    End Subroutine PbarHbarResFromOlap

    Subroutine PbarHbarEneLR(HOne,HTwo,Ref,Ene,T1,T2,Z1,Z2,NOcc,NAO,NSO,ENuc,QJ,SP, &
                             NPoints, Npg, ncisp, ncipg, ncik, R1, R2, Rpg, Rk, &
                             Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk,TrunODE)
    Implicit None
    Integer,           Intent(In)  :: NOcc, NAO, NSO
    Integer,           Intent(In)  :: NPoints(2), Npg
    Integer,           Intent(In)  :: TrunODE, QJ,SP
    Integer,           Intent(In)  :: ncisp, ncipg, ncik
    Real (Kind=pr),    Intent(In)  :: Roota(:), Rootb(:), Rooty(:)
    Real (Kind=pr),    Intent(In)  :: Weightsp(:,:), Weightpg(Npg,ncipg,ncipg)
    Complex (Kind=pr), Intent(In)  :: fsp(ncisp),fpg(ncipg),fk(ncik)
    Complex (Kind=pr), Intent(In)  :: R1(:,:,:), R2(:,:,:)
    Complex (Kind=pr), Intent(In)  :: Rpg(npg,NSO,NSO), Rk(ncik,NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: T1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr), Intent(In)  :: T2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    Complex (Kind=pr), Intent(In)  :: Z1(NOcc,NOcc+1:NSO)
    Complex (Kind=pr), Intent(In)  :: Z2(NOcc,NOcc,NOcc+1:NSO,NOcc+1:NSO)
    Complex (Kind=pr), Intent(In)  :: HOne(NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: HTwo(NSO,NSO,NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: Ref(NSO,NSO)
    Real (Kind=pr),    Intent(In)  :: ENuc 
    Complex (Kind=pr), Intent(Out) :: Ene 
    ! Allocation
    Integer                        :: IAlloc
    ! For excited Kernels
    Complex (Kind=pr)              :: Olap0, EOlap0
    Complex (Kind=pr), Allocatable :: OlapEx1(:,:), OlapEx2(:,:,:,:)
    Complex (Kind=pr), Allocatable :: EOlapEx1(:,:), EOlapEx2(:,:,:,:)
    ! LR variables
    Complex (Kind=pr)              :: Z0eff
    Complex (Kind=pr)              :: Numer, Denom 
    Complex (Kind=pr), Allocatable :: Z1eff(:,:), Z2eff(:,:,:,:)
    Integer   :: I, J, A, B

    !====================================================================!
    ! This is the driver for SGCCSD                                      !
    !====================================================================!
    Write(*,*) "Entered PbarHbarEneLR"
    Allocate(OlapEx1(NOcc+1:NSO,1:NOcc),                  &
             OlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc),    &
             EOlapEx1(NOcc+1:NSO,1:NOcc),                 &
             EOlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc),   &
             Z1eff(NOcc,NOcc+1:NSO),                      &
             Z2eff(NOcc,NOcc,NOcc+1:NSO,NOcc+1:NSO),      &
             Stat=IAlloc)
    If(IAlloc /= 0) Stop "Could not allocate in PbarHbarEneLR"
    Write(*,*) "TrunODE", TrunODE
    ! The coding of TrunODE is 
    ! 1: SD; SD
    ! 2: SDt; SD
    ! 3: SDt; SDtq
    ! 4: SDT; SDT
    ! 5: SDTq; SDT
    ! 6: SDTq; SDTq
    ! 7: SDTQ; SDTQ

    ! Carry out spin integration
    Call PbarHbarOlap(Olap0,OlapEx1,OlapEx2,EOlap0,EOlapEx1,&
                      EOlapEx2,HOne,HTwo,Ref,T1,T2,NOcc,NAO,NSO,ENuc,QJ,SP,&
                      NPoints, Npg, ncisp, ncipg, ncik, R1, R2, Rpg, Rk, &
                      Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk,TrunODE)
    ! Add linear response
    ! Build effective Z for CCSD Bra state
    !      Call BuildZeff(NOcc,NSO,T1,T2,Z1,Z2,Z0eff,Z1eff,Z2eff)
    !     Write(*,*) "Z0 eff ", Z0eff
    !     Call OutMat(*,NOcc,NSO-NOcc,2,Z1eff,"Z1eff")
    !     Call OutMat(*,NOcc**2,(NSO-NOcc)**2,2,Z2eff,"Z1eff")
    !     Call OutMat(*,NSO-NOcc,NOcc,2,OlapEx1,"Ovl1")
    !     Call OutMat(*,(NSO-NOcc)**2,NOcc**2,2,OlapEx2,"Ovl2")
    !     Call OutMat(*,NSO-NOcc,NOcc,2,EOlapEx1,"EOvl1")
    !     Call OutMat(*,(NSO-NOcc)**2,NOcc**2,2,EOlapEx2,"EOvl2")
    !      Numer = EOlap0 * Z0eff
    !      Denom = Olap0  * Z0eff
    !      Do I = 1, NOcc
    !      Do A = NOcc+1, NSO
    !        Numer = Numer + Z1eff(I,A)*EOlapEx1(A,I) 
    !        Denom = Denom + Z1eff(I,A)* OlapEx1(A,I) 
    !      EndDo
    !      EndDo
    !      Do I = 1, NOcc
    !      Do A = NOcc+1, NSO
    !      Do J = 1, NOcc
    !      Do B = NOcc+1, NSO
    !        Numer = Numer + Z2eff(I,J,A,B)*EOlapEx2(A,B,I,J)/Four 
    !        Denom = Denom + Z2eff(I,J,A,B)* OlapEx2(A,B,I,J)/Four 
    !      EndDo
    !      EndDo
    !      EndDo
    !      EndDo
    !     Write(*,*) "Original EOlap   ", EOlap0
    !     Write(*,*) "Original Olap    ",  Olap0
    Write(*,*) "Pbar Hbar Energy w/o response:", EOlap0/Olap0
    !      Write(*,*) "Pbar Hbar Energy w/  response:", Numer/Denom
    Ene = EOlap0/Olap0
    ! Deallocate
    DeAllocate(EOlapEx1, EOlapEx2, OlapEx1, OlapEx2, &
               Z1eff, Z2eff,  &
               Stat=IAlloc)
    Return
    End Subroutine PbarHbarEneLR

    Subroutine EvalKernel(HOne, HTwo, T1, T2, ENuc, &
                          eps0, lambda0, eps1, eps2, lambda1, lambda2, &
                          RotMat, NOcc, NAO, NSO, &
                          SP,TrunODE)
    Implicit None
    !------------------------------------------------------------------------------------------------
    ! Calculate components for PCCSD residuals, Eq 11
    !################################################################################################
    ! ==Inputs==
    ! NOcc : int
    !   Number of occupied spin orbitals
    ! NSO : int
    !   Number of spin oribtals
    ! NAO : int
    !   Number of atomic oribtals
    ! Enuc : real
    !   Energy of nuclear repulsion and core if any
    ! SP : int
    !   Parameter controlling if spin projection is done
    ! HOne : complex
    !   One electron integrals
    ! HTwo : complex
    !   Two electron integrals
    ! T1 : complex
    !   Current singles amplitudes
    ! T2 : complex
    !   Current doubles amplitudes
    ! RotMat : complex
    !   Collective projection operator
    ! TrunODE : int
    !   Value controlling excitation degree for U and W
    ! ===============================================================================================
    ! ==Outputs==
    ! eps0 : complex
    !   ?
    ! eps1 : complex
    !   ?
    ! eps2 : complex
    !   ?
    ! lambda0 : complex
    !   ?
    ! lambda1 : complex
    !   ?
    ! lambda2 : complex
    !   ?
    !------------------------------------------------------------------------------------------------
    Integer,           Intent(In)  :: NOcc, NAO, NSO
    Integer,           Intent(In)  :: TrunODE,SP
    Complex (Kind=pr), Intent(In)  :: RotMat(NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: T1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr), Intent(In)  :: T2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    Complex (Kind=pr), Intent(In)  :: HOne(NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: HTwo(NSO,NSO,NSO,NSO)
    Real (Kind=pr),    Intent(In)  :: ENuc
    Complex (Kind=pr), Intent(Out) :: eps0, lambda0
    Complex (Kind=pr), Intent(Out) :: eps1(NSO-NOcc,NOcc), eps2(NSO-NOcc,NSO-NOcc,NOcc,NOcc)
    Complex (Kind=pr), Intent(Out) :: lambda1(NSO-NOcc,NOcc), lambda2(NSO-NOcc,NSO-NOcc,NOcc,NOcc)
    Complex (Kind=pr)              :: Euler(3)
    Complex (Kind=pr)              :: OvlTh
    Complex (Kind=pr), Allocatable :: U1t(:,:), U2t(:,:,:,:)
    Complex (Kind=pr)              :: U0t, STot
    Complex (Kind=pr), Allocatable :: Fock(:,:), VR(:,:)
    Complex (Kind=pr), Allocatable :: eps1t(:,:), eps2t(:,:,:,:)
    Complex (Kind=pr), Allocatable :: lambda1t(:,:), lambda2t(:,:,:,:)
    Complex (Kind=pr)              :: ESCF, ET
    Integer                        :: IAlloc
    Integer         :: cnt1, cnt2, clock_rate, clock_max
    Allocate(Fock(NSO,NSO),    &
             U1t(NOcc+1:NSO,NOcc), U2t(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc),  &
             eps1t(NSO,NSO), lambda1t(NSO,NSO), &
             eps2t(NSO,NSO,NSO,NSO), lambda2t(NSO,NSO,NSO,NSO), &
             Stat=IAlloc)
    Call system_clock(cnt1,clock_rate,clock_max)

    Call DSTT2(RotMat,T1,T2,U0t,U1t,U2t,NOcc,NSO)

    Call system_clock(cnt2,clock_rate,clock_max)
    ! Write(*,*) 'Trans T2 time = ', real(cnt2-cnt1)/real(clock_rate)

    !      Call BuildUtildeSDFCI(V1,T1,T2,U0t,U1t,U2t,NOcc,NAO,NSO)
    STot = U0t
    Call MKFock_MO(HOne,HTwo,Fock,ESCF,NOcc,NSO)
    ESCF = ESCF + ENuc
    Call system_clock(cnt1,clock_rate,clock_max)
    If ((Abs(TrunODE).ge.1).and.(Abs(TrunODE).le.3)) then
      Call BuildKernSD(lambda1t,lambda2t,eps0,eps1t,eps2t,   &
          Fock,HTwo,ESCF,U1t,U2t,T1,T2,NOcc,NSO,SP,Abs(TrunODE))
    !      ElseIf (TrunODE.eq.0) Then
    !        Call BuildExactKern(STot,lambda1t,lambda2t,eps0,eps1t,eps2t,V1,  &
    !            HOneV1,HTwoV1,T1,T2,NDet,NOcc,NOcc/2,NOcc/2,NAO,NSO)
    !        STot = OvlTh * STot
    EndIf
    If (TrunODE .eq. 3) then
        Call CCSD_T_Ene(Fock, HTwo, U1t, U2t, NOcc, NSO-NOcc, NSO, ET)
        eps0 = eps0 + ET
        Write(*,*) "(T) Ene =", ET
    End If
    Deallocate(Fock, U1t, U2t)
    Call system_clock(cnt2,clock_rate,clock_max)
    ! Write(*,*) 'CC time = ', real(cnt2-cnt1)/real(clock_rate)
    if (SP == 2) then
      Call ProjSz1(lambda1t,NOcc,NAO,NSO)
      Call ProjSz1(eps1t,NOcc,NAO,NSO)
      Call ProjSz2(lambda2t,NOcc,NAO,NSO)
      Call ProjSz2(eps2t,NOcc,NAO,NSO)
    end if
    lambda0 = STot
    eps0    = eps0*lambda0
    lambda1 = lambda1t(NOcc+1:NSO,:NOcc)*lambda0
    eps1    = eps1t(NOcc+1:NSO,:NOcc)*lambda0
    lambda2 = lambda2t(NOcc+1:NSO,NOcc+1:NSO,:NOcc,:NOcc)*lambda0
    eps2    = eps2t(NOcc+1:NSO,NOcc+1:NSO,:NOcc,:NOcc)*lambda0
    !      Write(*,*) " W0                   ", U0t
    !      Write(*,*) "<beta|0>              ", OvlTh
    !      Write(*,*) "<beta| exp(U1+U2)|0>  ", lambda0
    !      Write(*,*) "<beta| H exp(U1+U2)|0>", eps0
    Deallocate(eps1t,lambda1t,eps2t,lambda2t)
    End Subroutine

    Subroutine EvalOvlp(C1,C2,Ovlp0, Ovlp1,Ovlp2,RotMat, NOcc,NSO,SP)
    Implicit None
    Integer,           Intent(In)  :: NOcc, NSO,SP
    Complex (Kind=pr), Intent(In)  :: RotMat(NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: C1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr), Intent(In)  :: C2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    Complex (Kind=pr), Intent(Out) :: Ovlp0, Ovlp1(NSO,NSO)
    Complex (Kind=pr), Intent(Out) :: Ovlp2(NSO,NSO,NSO,NSO)
    Complex (Kind=pr), Allocatable :: V1(:,:), VR(:,:)
    Complex (Kind=pr), Allocatable :: C1V1(:,:), C2V1(:,:,:,:)
    Complex (Kind=pr), Allocatable :: Fock(:,:)
    Complex (Kind=pr) :: ESCF
    Integer        :: i, j, a, b, p, q, r, s, NAO
    Integer                        :: IAlloc
    Allocate(V1(NOcc,NOcc+1:NSO), VR(NSO,NSO), &
             C1V1(NSO,NSO), C2V1(NSO,NSO,NSO,NSO),     &
             Fock(NSO,NSO), Stat=IAlloc)
    NAO = NSO / 2
    C1V1 = Zero
    C2V1 = Zero
    Ovlp1 = Zero
    Ovlp2 = Zero
    C1V1(NOcc+1:NSO,:NOcc) = C1
    C2V1(NOcc+1:NSO,NOcc+1:NSO,:NOcc,:NOcc) = C2
    Call BuildV1(V1,Ovlp0,RotMat,NOcc,NSO)
    Call IntTranDeex2(C1V1,V1,NOcc,NSO)
    Call IntTranDeex4(C2V1,V1,NOcc,NSO)
    Call MKFock_MO(C1V1,C2V1,Fock,ESCF,NOcc,NSO)
    Ovlp1(NOcc+1:NSO,:NOcc) = Fock(NOcc+1:NSO,:NOcc)
    !$omp parallel do
    Do I = 1, NOcc
      Ovlp1(I,I) = ESCF
    End Do
    !$omp end parallel do
    if (SP == 2) then
      Call ProjSz1(Ovlp1, NOcc, NAO, NSO)
    endif
    Ovlp2(NOcc+1:NSO,NOcc+1:NSO,:NOcc,:NOcc) = C2V1(NOcc+1:NSO,NOcc+1:NSO,:NOcc,:NOcc)
    !$omp parallel do
    Do I = 1, NOcc
    Do J = 1, NOcc
      If(J.ne.I) then
      Ovlp2(I,J,I,J) =  ESCF
      Ovlp2(I,J,J,I) = -ESCF
      Do A = NOcc+1, NSO
           Ovlp2(A,J,I,J) =  Ovlp1(A,I)
           Ovlp2(J,A,J,I) =  Ovlp1(A,I)
           Ovlp2(J,A,I,J) = -Ovlp1(A,I)
           Ovlp2(A,J,J,I) = -Ovlp1(A,I)
      End Do
      End If
    End Do
    End Do
    !$omp end parallel do
    if (SP == 2) then
      Call ProjSz2(Ovlp2, NOcc, NAO, NSO)
    endif
    Call IDMat(VR, NSO)
    VR(:NOcc,NOcc+1:NSO) = -V1
    VR = MatMul(RotMat,VR)
    Call IntTran2Sim( Ovlp1,VR, NSO)
    Call IntTran4Sim( Ovlp2,VR, NSO)
    if (SP == 2) then
      Call ProjSz1(Ovlp1,NOcc,NAO,NSO)
      Call ProjSz2(Ovlp2,NOcc,NAO,NSO)
    end if
    Deallocate(V1,VR,C1V1,C2V1,Fock)
    End Subroutine


    Subroutine WriteW(U1,U2,NOcc,NSO,N)
    Implicit None
    Integer,           Intent(In)  :: NOcc, NSO
    Integer,           Intent(In)  :: N
    Complex (Kind=pr), Intent(In)  :: U1(NOcc+1:NSO,NOcc)
    Complex (Kind=pr), Intent(In)  :: U2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
    character(len=20) :: str
    write (str,*) N
    str = adjustl(str)
    open(11, file='GridT1_'//trim(str))
    open(12, file='GridT2_'//trim(str))
    write (11, *) U1
    write (12, *) U2
    close (11)
    close (12)
    End Subroutine

    Subroutine WriteH(H1,H2,NSO,N)
    Implicit None
    Integer,           Intent(In)  :: NSO
    Integer,           Intent(In)  :: N
    Complex (Kind=pr), Intent(In)  :: H1(NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: H2(NSO,NSO,NSO,NSO)
    character(len=20) :: str
    write (str,*) N
    str = adjustl(str)
    open(11, file='H1_'//trim(str), form="unformatted")
    open(12, file='H2_'//trim(str), form="unformatted")
    write (11) H1
    write (12) H2
    close (11)
    close (12)
    End Subroutine

    Subroutine ReadH(H1,H2,NSO,N)
    Implicit None
    Integer,           Intent(In)  :: NSO
    Integer,           Intent(In)  :: N
    Complex (Kind=pr), Intent(In)  :: H1(NSO,NSO)
    Complex (Kind=pr), Intent(In)  :: H2(NSO,NSO,NSO,NSO)
    character(len=20) :: str
    write (str,*) N
    str = adjustl(str)
    open(11, file='H1_'//trim(str), form="unformatted")
    open(12, file='H2_'//trim(str), form="unformatted")
    write (11) H1
    write (12) H2
    close (11)
    close (12)
    End Subroutine


End Module FPUCC_nompi
