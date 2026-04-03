Module DIIS

  Use Precision
  Use Constants
  Use Update_nompi
  Implicit None
  Public DIIS_PCC
  ! Integer,        Parameter   :: NDIIS = 5
  ! Integer,        Parameter   :: LenV  = NDIIS+1
  ! Integer,        Parameter   :: StartDIIS = 15
  ! Real (Kind=pr), Allocatable :: ErrVecs(:,:)
  ! Real (Kind=pr) :: M(LenV,LenV)
  ! Real (Kind=pr) :: V(LenV)
 
  Contains
  
    Subroutine DIIS_PCC(Res, Ene, VecsIn,ErrsIn,NVecs,VecLen,thrsh, &
      HOne,HTwo,Ref,NOcc,NAO,NSO,ENuc,QJ, SP, &
      NPoints,Npg, ncisp, ncipg, ncik, R1, R2, Rpg, Rk, &
      Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk,TrunODE)
      !------------------------------------------------------------------------------------------------
      ! Perform PUCC optimization with DIIS acceleration
      !################################################################################################
      ! ==Inputs==
      ! NSO : int
      !   Number of spin oribtals
      ! HTwo : complex
      !   Two electron integrals
      ! Vecs : complex
      !   Carry Error vectors(the residuals)
      ! NVec : int
      !   Current number of Error vectors
      ! VecLen : int
      !   Length of each error vector
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
   
      Integer,           Intent(In)  :: NVecs, VecLen
      Complex (Kind=pr), Intent(In)  :: VecsIn(Nvecs*VecLen),ErrsIn(Nvecs*VecLen)
      Integer,           Intent(In)  :: NOcc, NAO, NSO
      Integer,           Intent(In)  :: NPoints(2), Npg
      Integer,           Intent(In)  :: TrunODE, QJ, SP
      Integer,           Intent(In)  :: ncisp, ncipg, ncik
      Real (Kind=pr),    Intent(In)  :: Roota(:), Rootb(:), Rooty(:)
      Real (Kind=pr),    Intent(In)  :: Weightsp(:,:), Weightpg(:,:,:)
      Complex (Kind=pr), Intent(In)  :: fsp(ncisp),fpg(ncipg),fk(ncik)
      Complex (Kind=pr), Intent(In)  :: R1(:,:,:), R2(:,:,:)
      Complex (Kind=pr), Intent(In)  :: Rpg(npg,NSO,NSO), Rk(ncik,NSO,NSO)
      Complex (Kind=pr), Intent(In)  :: HOne(NSO,NSO)
      Complex (Kind=pr), Intent(In)  :: HTwo(NSO,NSO,NSO,NSO)
      Complex (Kind=pr), Intent(In)  :: Ref(NSO,NSO)
      Real (Kind=pr),    Intent(In)  :: ENuc 
      double precision,  Intent(In) :: thrsh
      Complex (Kind=pr), Intent(Out)  :: Res(VecLen)
      Complex (Kind=pr), Intent(Out)  :: Ene
   
      integer :: i,j,a,d,n,info,lwork,NVec
      integer :: MaxCyc = 20
      double precision,dimension(:,:),allocatable :: B, B_
      double precision,dimension(:),allocatable :: y,c,work,p_
      integer,dimension(:),allocatable :: iwork
      complex(kind=pr),dimension(:),allocatable,target :: err1,err2,x1,x2
      complex(kind=pr),dimension(:),allocatable :: errnew,xnew
      complex(kind=pr),dimension(:),pointer :: err,x
      complex(kind=pr) :: EneOld
      integer,dimension(:),allocatable :: ipiv
      double precision :: E_old, anorm, temp, error, rcond

      NVec = NVecs
      EneOld = Zero
   
      allocate(xnew(VecLen))
      allocate(errnew(VecLen))
      allocate(x1(Nvec*VecLen))
      allocate(err1(Nvec*VecLen))
      call zcopy(Nvec*VecLen, VecsIn, 1, x1, 1)
      call zcopy(Nvec*VecLen, ErrsIn, 1, err1, 1)

      do n = 1, MaxCyc

        if (mod(n, 2) == 1) then
          x => x1
          err => err1
        else
          x => x2
          err => err2
        endif

        if ((n == 1) .and. (Nvec == 1)) then
          ! No need to build B if only 1 entry vector
          print *, "Need to implement direct update to PCC"
          stop
          x => x1
        else
          ! Establish DIIS matrix and vector
          allocate(y(NVec+1))
          y = Zero
          y(NVec+1) = 1

          allocate(B(NVec+1, NVec+1))
          B = One
          B(NVec+1,NVec+1) = 0
          do i = 1, NVec
            do j = i, NVec
              B(i, j) = zabs(DOT_PRODUCT(err((i-1)*VecLen + 1:i*VecLen), err((j-1)*VecLen + 1:j*VecLen)))
              if (i /= j) then
                B(j, i) = B(i, j)
              endif
            enddo
          enddo
        endif
        do i = 1, NVec
          print *, B(i,i)
          ! print *, norm2(abs(err((i-1)*VecLen + 1:i*VecLen)))
        enddo
        ! stop

        !Check condition number of B

        ! Calculate 1-norm of B
        anorm = Zero
        do i = 1, Nvec + 1
          temp = sum(abs(B(:,i)))
          if (temp > anorm) then
            anorm = temp
          endif
        enddo
        allocate(work(4*(Nvec+1)))
        allocate(iwork(Nvec+1))
        call dgecon("1", NVec+1, B, Nvec+1, anorm, rcond, work, iwork, info)
        if (info /= 0) then
          print *, "Failed to check condition number of B"
          stop
        endif
        deallocate(work)
        deallocate(iwork)

        ! if (rcond < 1.0D-15) then
        if (.false.) then
          ! Use last vector instead of DIIS
          print *, n
          print *, rcond, "Need to implement direct update to PCC"
          stop
        else
          ! Solve DIIS linear problem for next guess, solution placed in y
          allocate(B_(NVec+1, NVec+1))
          Call dcopy((NVec+1)*(NVec+1), B, 1, B_, 1)
          allocate(ipiv(Nvec+1))
          call dgesv(Nvec+1, 1, B, NVec+1, ipiv, y, NVec+1, info)
          if (info /= 0) then
            stop "MATRIX INVERSION FAILED!!!"
          endif
          deallocate(ipiv)

          ! Approximate error
          error = Dot_Product(y(1:Nvec), MatMul(B_(1:Nvec, 1:Nvec), y(1:NVec)))
          print *, "Current DIIS Error", error
          ! error = B(Nvec,Nvec)
          deallocate(B_)

          ! Build new amplitudes
          xnew = Zero
          do i = 1, Nvec
            ! xnew = xnew + y(i)*x((i-1)*VecLen + 1:i*VecLen)
            Call zaxpy(VecLen, dcmplx(y(i)), x((i-1)*VecLen + 1:i*VecLen), 1, xnew)
          enddo

          ! ! Check for convergence
          ! if (dsqrt(error) < thrsh) then
          !   call zcopy(VecLen, xnew, 1, Res, 1)
          !   deallocate(xnew,errnew)
          !   print *, "Converged!!!"
          !   exit
          ! endif
          ! print *, "Current Error", dsqrt(abs(error)), error

          ! Get new residual
          Call EvalF_(xnew,errnew,Ene,VecLen, &
            HOne,HTwo,Ref,NOcc,NAO,NSO,ENuc,QJ, SP, &
            NPoints,Npg, ncisp, ncipg, ncik, R1, R2, Rpg, Rk, &
            Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk,TrunODE)
          Call BuildSRes_(errnew,VecLen, &
            HOne,HTwo,Ref,NOcc,NAO,NSO,ENuc,QJ, SP, &
            NPoints,Npg, ncisp, ncipg, ncik, R1, R2, Rpg, Rk, &
            Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk,TrunODE)

          print *, Ene

          error = real(abs(Ene - EneOld))
          print *, "Current Energy Error", error
          error = norm2(zabs(errnew))
          print *, "Current res Error", error

          ! Check for convergence
          if (error < thrsh) then
            call zcopy(VecLen, xnew, 1, Res, 1)
            deallocate(xnew,errnew)
            print *, "Converged!!!"
            exit
          endif

          EneOld = Ene

          ! Maybe remove old guesses?

          ! Extend x and err
          NVec = Nvec + 1
          if (mod(n, 2) == 1) then
            allocate(x2(Nvec*VecLen))
            allocate(err2(Nvec*VecLen))
            call zcopy((Nvec-1)*VecLen, x1, 1, x2, 1)
            call zcopy((Nvec-1)*VecLen, err1, 1, err2, 1)
            call zcopy(VecLen, xnew, 1, x2((Nvec-1)*VecLen + 1), 1)
            call zcopy(VecLen, errnew, 1, err2((Nvec-1)*VecLen + 1), 1)
            deallocate(x1)
            deallocate(err1)
          else
            allocate(x1(Nvec*VecLen))
            allocate(err1(Nvec*VecLen))
            call zcopy((Nvec-1)*VecLen, x2, 1, x1, 1)
            call zcopy((Nvec-1)*VecLen, err2, 1, err1, 1)
            call zcopy(VecLen, xnew, 1, x1((Nvec-1)*VecLen + 1), 1)
            call zcopy(VecLen, errnew, 1, err1((Nvec-1)*VecLen + 1), 1)
            deallocate(x2)
            deallocate(err2)
          endif


        endif

        deallocate(y)
        deallocate(B)
      enddo

    End Subroutine DIIS_PCC

    ! Subroutine SetUpDIIS(LenVec)
    ! Implicit None
    ! Integer, Intent(In) :: LenVec
    ! Allocate(ErrVecs(LenVec,NDIIS))
 
    ! End Subroutine SetUpDIIS
 
    ! Subroutine DoDIIS(Coeffs)
    ! Implicit None
    ! Real (Kind=pr), Intent(Out) :: Coeffs(NDIIS)
    ! Integer :: Info, IPiv(LenV)
    ! Integer :: I, J
 
    ! !========================================================!
    ! !  Given error vectors ErrVecs, set up the DIIS problem  !
    ! !  and get the coefficient array Coeffs.                 !
    ! !========================================================!
    
    ! ! Initialize V and M
    ! V = Zero
    ! M = One
    ! V(LenV) = One
    ! M(LenV,LenV) = Zero
 
 
    ! ! Build the B-matrix part of M
    ! Do I = 1,NDIIS
    ! Do J = 1,NDIIS
    !     M(I,J) = Dot_Product(ErrVecs(:,I),ErrVecs(:,J))
    ! End Do
    ! End Do
 
 
    ! ! Solve M.C = V, placing the solution in V
    ! Call DGeSV(LenV,1,M,LenV,IPiv,V,LenV,Info)
    ! If(Info == 0) Coeffs = V(1:NDIIS)
 
    ! Return
    ! End Subroutine DoDIIS
 
    ! Subroutine ShutDownDIIS
    ! Implicit None
    ! DeAllocate(ErrVecs)
    ! Return
    ! End Subroutine ShutDownDIIS
End Module DIIS

