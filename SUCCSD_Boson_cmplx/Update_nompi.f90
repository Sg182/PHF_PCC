Module Update_nompi
  Use Precision
  Use Constants
  ! Use UHF
  Use FPUCC_nompi
  Use FPUCC_Tools
  Implicit None
  Private
  Public  :: EvalF_, BuildSRes_

  Real (Kind=pr)                 :: damp = 1.0_pr
  Logical                        :: DoCCD = .False.

  Contains
    Subroutine EvalF_(x, fx,Ene, nBrd, &
      HOne,HTwo,Ref,NOcc,NAO,NSO,ENuc,QJ, SP, &
      NPoints,Npg, ncisp, ncipg, ncik, R1, R2, Rpg, Rk, &
      Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk,TrunODE)

      Complex (Kind=pr), Intent(In)  :: x(nBrd)
      Integer,           Intent(In)  :: NOcc, NAO, NSO, nBrd
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
      Complex (Kind=pr), Intent(Out) :: fx(nBrd), Ene
      Complex (Kind=pr)              :: T1(NOcc+1:NSO,NOcc), T2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
      Complex (Kind=pr)              :: Res1(NOcc+1:NSO,NOcc), Res2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
      Complex (Kind=pr)              :: OlapEx1(NOcc+1:NSO,NOcc), OlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
      Complex (Kind=pr)              :: EOlapEx1(NOcc+1:NSO,NOcc), EOlapEx2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
      Complex (Kind=pr)              :: Olap0, EOlap0
      Complex (Kind=pr)              :: Res0
      Complex (Kind=pr)              :: Norm1, Norm2, scal


      ! Determine what T1 and T2 are
      Call Vec2Mat_(x,Ene,T1,T2,NSO,NOcc,nBrd)
      ! Build Residuals 
      If(DoCCD) T1 = Zero ! CCD!
      Call PbarHbarOlap(Olap0,OlapEx1,OlapEx2,EOlap0,EOlapEx1,  &
                        EOlapEx2,HOne,HTwo,Ref,T1,T2,  &
                        NOcc,NAO,NSO,ENuc,QJ,SP,  &
                        NPoints, Npg, ncisp, ncipg, ncik, &
                        R1, R2, Rpg,Rk,Roota,Rootb,Rooty, &
                        Weightsp,Weightpg,fsp,fpg,fk,TrunODE)

      Call PbarHbarResFromOlap(Ene,Res0,Res1,Res2,  &
           Ref,Olap0,OlapEx1,OlapEx2,EOlap0,&
           EOlapEx1,EOlapEx2,NOcc,NAO,NSO,SP)
      ! EneBrd = Ene

      If(DoCCD) Res1 = Zero ! CCD!

      ! Put dUtilde into dy
      Call Mat2Vec_(fx,Res0,Res1,Res2,NSO,NOcc,nBrd)

    End Subroutine EvalF_

    Subroutine BuildSRes_(Res,nBrd, &
      HOne,HTwo,Ref,NOcc,NAO,NSO,ENuc,QJ, SP, &
      NPoints,Npg, ncisp, ncipg, ncik, R1, R2, Rpg, Rk, &
      Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk,TrunODE)
      Integer,           Intent(In)  :: NOcc, NAO, NSO, nBrd
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
      Complex (Kind=pr), Intent(InOut)  :: Res(nBrd)
      Complex (Kind=pr)              :: r0tmp, r1tmp(NOcc+1:NSO,NOcc)
      Complex (Kind=pr)              :: r2tmp(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
      Complex (Kind=pr)              :: t1(NOcc+1:NSO,NOcc)
      Complex (Kind=pr)              :: t2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
      Complex (Kind=pr)              :: tmp
      Integer :: i,j,a,b
      tmp = Res(1)
      Call Vec2Mat_(Res,r0tmp,r1tmp,r2tmp,NSO,NOcc,nBrd)
      Call BuildSC_test(t1,t2,r1tmp,r2tmp,NOcc,NSO,NPoints,Npg,QJ,SP, &
                        ncisp,ncipg,ncik,R1, R2, Rpg, Rk, &
                        Roota,Rootb,Rooty,Weightsp,Weightpg,fsp,fpg,fk)
      Call Mat2Vec_(Res,tmp,t1,t2,NSO,NOcc,nBrd)
      Res = Res / damp
    End Subroutine BuildSRes_

    Subroutine Mat2Vec_(y,Ene,T1,T2,NSO,NOcc,nBrd)
      Integer,   Intent(In)            :: NSO, NOcc, nBrd
      Complex (Kind=pr),   Intent(In)  :: Ene, T1(NOcc+1:NSO,NOcc)
      Complex (Kind=pr),   Intent(In)  :: T2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
      Complex (Kind=pr),   Intent(Out) :: y(nBrd)
      Integer       :: IamNum
      Integer       :: I, J, A, B
      IamNum = 0
      ! Put Ene into y
      IamNum = IamNum + 1
      y(IamNum) = Ene 
      ! Put T1 into y
      Do I = 1, NOcc
      Do A = NOcc+1, NSO
        IamNum    = IamNum + 1
        y(IamNum) = T1(A,I)
      EndDo
      EndDo
      ! Put T2tilde into y
      Do J = 2, NOcc
      Do I = 1, J-1 
      Do B = NOcc+2, NSO
      Do A = NOcc+1, B-1 
        IamNum    = IamNum + 1
        y(IamNum) = T2(A,B,I,J)
      EndDo
      EndDo
      EndDo
      EndDo
      ! Check compatibility between dimension of y and Utildes
      If(nBrd.ne.IamNum) Stop "Dimension of y is not compatible within Broyden"
    End Subroutine Mat2Vec_

    Subroutine Vec2Mat_(y,Ene,T1,T2,NSO,NOcc,nBrd)
      Integer,   Intent(In)            :: NSO, NOcc, nBrd
      Complex (Kind=pr),   Intent(In)  :: y(nBrd)
      Complex (Kind=pr),   Intent(Out) :: Ene, T1(NOcc+1:NSO,NOcc)
      Complex (Kind=pr),   Intent(Out) :: T2(NOcc+1:NSO,NOcc+1:NSO,NOcc,NOcc)
      Complex (Kind=pr)    :: tmp 
      Integer              :: IamNum
      Integer              :: I, J, A, B
      IamNum = 0
      ! Get Ene from y
      IamNum = IamNum + 1
      Ene = y(IamNum) 
      ! Get T1 from y
      Do I = 1, NOcc
      Do A = NOcc+1, NSO
        IamNum  = IamNum + 1
        T1(A,I) = y(IamNum) 
      EndDo
      EndDo
      ! Get T2 from y
      T2 = Zero
      Do J = 2, NOcc
      Do I = 1, J-1 
      Do B = NOcc+2, NSO
      Do A = NOcc+1, B-1 
        IamNum = IamNum + 1
        tmp    = y(IamNum)
        T2(A,B,I,J) =  tmp
        T2(B,A,J,I) =  tmp
        T2(A,B,J,I) = -tmp
        T2(B,A,I,J) = -tmp
      EndDo
      EndDo
      EndDo
      EndDo
      ! Check compatibility between dimension of y and Utildes
      If(nBrd.ne.IamNum) Stop "Dimension of y is not compatible within Broyden "
    End Subroutine Vec2Mat_

End Module Update_nompi

