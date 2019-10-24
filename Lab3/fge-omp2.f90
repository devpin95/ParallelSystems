!-----------------------------------------------------------
! Gaussian elimination program : Fortran OpenMP Version 2
!-----------------------------------------------------------
!  Some features:
!    + Columnwise Data layout 
!    + Columnwise Elimination
!    + Static job scheduling 
!    + Parallel region
!    + Synchonization using critial section
!  Programming by: Gita Alaghband, Lan Vu 
!-----------------------------------------------------------
! Get user input of matrix dimension and printing option
!-----------------------------------------------------------
      SUBROUTINE GetUserInput(n,numThreads,isPrint,isOK)
      INTEGER :: argc
      CHARACTER(10) :: argv
      INTEGER,INTENT(OUT):: n,isPrint,numThreads
      LOGICAL,INTENT(OUT) :: isOK
      
      isOK = .TRUE.
      argc = iargc() 
      IF (argc < 2) THEN
           PRINT *,"Arguments:<X> <Y> [<Z>]"        
           PRINT *,"X : Matrix size [X x X]"              
           PRINT *,"Y : Number of Threads"
           PRINT *,"Z = 1: print the input/output matrix if X < 10"              
           PRINT *,"Z <> 1 or missing: does not print the input/output matrix"         
           isOK = .FALSE.     
      ELSE
           ! get matrix size argument     
         
           call getarg(1, argv)
           read (argv,*) n
           IF (n <= 0 ) THEN
                PRINT *,"Matrix size must be larger than 0"         
                isOK = .FALSE.     
           ENDIF
                                 			
           ! get number of threads argument          
         
           call getarg(2, argv)
           read (argv,*) numThreads 
           IF (numThreads <= 0) THEN
                PRINT *,"Number of threads must be larger than 0"         
                isOK = .FALSE.     
           ENDIF
           
  		   ! is print the input/output matrix argument
           isPrint = 0
           IF (argc >= 3) THEN
                call getarg(3, argv)
                read (argv,*) isPrint            
                IF (isPrint == 1 .AND. n <= 9) THEN
                     isPrint = 1
                ELSE
                     isPrint = 0
                ENDIF
           ENDIF     
      ENDIF  
      RETURN
      END
!----------------------------------------------------------------
!  Initialize the value of matrix a[n x n] for Gaussian elimination  
!----------------------------------------------------------------
      SUBROUTINE InitializeMatrix(a, n)
      USE omp_lib
      INTEGER :: i,j,n
      REAL :: a(n,n)

      !$OMP PARALLEL DO PRIVATE(i,j) SCHEDULE(STATIC) 
      DO i = 1,n 	  		
           DO j = 1,n
                IF (i == j) THEN
                     a(j,i) = i*i/2.	
                ELSE
                     a(j,i) = (i+j)/2.
                ENDIF
           ENDDO
      ENDDO

      RETURN
      END SUBROUTINE InitializeMatrix
!----------------------------------------------------------------
!  Print matrix  
!----------------------------------------------------------------
      SUBROUTINE PrintMatrix(a, n)
      INTEGER :: i,j,n
      REAL :: a(n,n)
      DO i = 1,n 	  		
           WRITE (*,'(a,i1,a,$)') "Row ",i,":"
           DO j = 1,n 	  		
                WRITE (*,'(F8.2,$)')  a(j,i)
           ENDDO 

           ! go to new line
           PRINT *,""
      ENDDO
      RETURN
      END SUBROUTINE PrintMatrix
!----------------------------------------------------------------
! Compute the Gaussian Elimination for matrix a[n x n]
!----------------------------------------------------------------
      SUBROUTINE ComputeGaussianElimination(a,n,isOK)
      USE omp_lib
      REAL :: pivot,gmax,pmax,temp
      INTEGER :: pindmax,gindmax,i,j,k,n
      REAL :: a(n,n)
      LOGICAL , INTENT(OUT) :: isOK
      isOK = .TRUE.

      !$OMP PARALLEL SHARED(a,gmax,gindmax) FIRSTPRIVATE(n,k) PRIVATE(pivot,i,j,temp,pmax,pindmax)

           ! Perform columnwise elimination
           DO k = 1,n-1
          
                !$OMP SINGLE
                gmax = 0.0
                !$OMP END SINGLE
                pmax = 0.0;

                ! Find the pivot column
                ! Each thread works on a number of columns to find the local max value pmax
                ! Then update this max local value to the global variable gmax

                !$OMP DO SCHEDULE(static) 
                DO i = k,n
                    temp = abs(a(k,i))
                    IF (temp > pmax) THEN
                       pmax = temp;
                       pindmax = i;
                    ENDIF
                ENDDO

                
                ! gmax is updated one by one
                !$OMP CRITICAL
                IF (gmax < pmax) THEN
                    gmax = pmax
                    gindmax = pindmax
                ENDIF
                !$OMP END CRITICAL

                ! All threads have to reach this point before continue
                !$OMP BARRIER

                ! If matrix is singular set the flag & quit
     	        IF (gmax == 0) THEN 
     	             isOK = .FALSE.
     	             EXIT
     	        ENDIF     

                ! Swap columns if necessary
                IF (gindmax /= k) THEN
           
                    !$OMP DO SCHEDULE(static) 
                     DO j = k,n
                         temp = a(j,gindmax)
                         a(j,gindmax) = a(j,k)
                         a(j,k) = temp
                     ENDDO
                ENDIF

                ! Compute the pivot
                pivot = -1.0/a(k,k)

                ! Perform columnwise reductions
 
                !$OMP DO SCHEDULE(static) 
                DO i = k+1,n
                     temp = pivot*a(k,i)
                     a(k:n,i) = a(k:n,i) + temp*a(k:n,k)
                ENDDO
           ENDDO  
      !$OMP END PARALLEL
          
      RETURN
      END SUBROUTINE  ComputeGaussianElimination
!-----------------------------------------------------------
!  Main program
!-----------------------------------------------------------
      PROGRAM GaussianElimination
      USE omp_lib
      IMPLICIT NONE
      REAL, ALLOCATABLE, DIMENSION(:,:) :: a
      INTEGER :: n,isPrintMatrix,numThreads
      LOGICAL :: isOK
      DOUBLE PRECISION :: runtime

      CALL GetUserInput(n,numThreads,isPrintMatrix,isOK)
      
      IF (isOK .eqv. .FALSE.) GOTO 200

      ! Specify number of threads that program run with
      CALL OMP_SET_NUM_THREADS(numThreads)

      ALLOCATE(a(n,n))

      ! Initialize the value of matrix a[n x n]
      CALL InitializeMatrix(a,n)

      IF (isPrintMatrix == 1) THEN
           PRINT *,"Input matrix:"           
           CALL PrintMatrix(a, n)
      ENDIF

      runtime = omp_get_wtime()

      ! Compute the Gaussian Elimination for matrix a[n x n]
      CALL ComputeGaussianElimination(a,n,isOK)

      runtime = omp_get_wtime() - runtime
	
      IF (isOK .EQV. .FALSE.) THEN 
           PRINT *,"The matrix is singular"
      ELSE
           IF (isPrintMatrix == 1) THEN
                PRINT *,"Output matrix:"           
                CALL PrintMatrix(a, n)
           ENDIF
                WRITE (*,'(A,F0.2,A)') "Gaussian Elimination runs in ",runtime, " seconds" 
      ENDIF

      DEALLOCATE(a)

200   STOP
      END
!----------------------------------------------------------------

