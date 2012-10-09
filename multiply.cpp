#include "multiply.h"
#include <math.h>
#include <omp.h>

// at the moment, we assume that there are enough powers of 2 and 7 in the matrix dimension that no padding / uneven distribution is needed

char *pattern = NULL;

void multiply( double *A, double *B, double *C, MatDescriptor desc, char *patt ) {
  // allocate the memory we'll need for the eventual 1proc multiplies
  omp_set_num_threads(NUM_THREADS);
  int size1p;
  double *work = NULL;
  size1p = desc.lda/(1<< desc.nrec);
  if( desc.bs != 1 )
    work = allocate(size1p*size1p*3);
  else if ( desc.nproc_summa != 1 )
    work = allocate(size1p*size1p*3*2/desc.nproc_summa);
  pattern = patt;
  multiplyInternal( A, B, C, desc, work );
  if( desc.bs != 1 )
    deallocate(work, size1p*size1p*3);
  else if( desc.nproc_summa != 1 )
    deallocate(work, size1p*size1p*3*2/desc.nproc_summa);
}

// nproc is the number of processors that share the matrices, and will be involved in the multiplication
void multiplyInternal( double *A, double *B, double *C, MatDescriptor desc, double *work ) {
  if( desc.nrec == 0 ) { // (planned) out of recursion in the data layout, do a regular matrix multiply.  The matrix is now in a 2d block cyclic layout
    if( desc.nproc == 1 ) {
      if( desc.nproc_summa == 1 ) {
	// A 2d block cyclic layout with 1 processor still has blocks to deal with
	// run a 1-proc non-strassen
	block_multiply( A, B, C, desc, work );
      } else {
	summa1d( A, B, C, desc, work );
      } 
    } else {
      printf("Need more levels of recursion for this many procs?\n");
      assert( false );
    }
  } else {
    if( pattern == NULL ) {
      bool enoughMem = enoughMemory( SEVEN*numEntriesPerProc(desc)/4 );
      if( (desc.nproc > 1 && enoughMem ) || (powl(7,desc.nrec) == desc.nproc) ) {
	setExecutionType(desc.nrec, "BFS");
	strassenBFS( A, B, C, desc, work );
      } else if( desc.nproc == 1 && desc.nproc_summa == 6 && enoughMem ) {
	setExecutionType(desc.nrec, "HYB");
	strassenHybrid6( A, B, C, desc, work );
      } else if( desc.nproc == 1 && desc.nproc_summa == 5 && enoughMem ) {
	setExecutionType(desc.nrec, "HYB");
	strassenHybrid5( A, B, C, desc, work );
      } else if( desc.nproc == 1 && desc.nproc_summa == 4 && enoughMem ) {
	setExecutionType(desc.nrec, "HYB");
	strassenHybrid4( A, B, C, desc, work );
      } else if( desc.nproc == 1 && desc.nproc_summa == 3 && enoughMem ) {
	setExecutionType(desc.nrec, "HYB");
	strassenHybrid3( A, B, C, desc, work );
      } else if( desc.nproc == 1 && desc.nproc_summa == 2 && enoughMem ) {
	setExecutionType(desc.nrec, "HYB");
	strassenHybrid2( A, B, C, desc, work );
      } else {
	setExecutionType(desc.nrec, "DFS");
	strassenDFS( A, B, C, desc, work );
      }
    } else {
      if( pattern[0] == 'B' || pattern[0] == 'b' ) {
	setExecutionType(desc.nrec, "BFS");
	pattern++;
	strassenBFS( A, B, C, desc, work );
	pattern--;
      } else if( pattern[0] == 'D' || pattern[0] == 'd' ) {
	setExecutionType(desc.nrec, "DFS");
	pattern++;
	strassenDFS( A, B, C, desc, work );
	pattern--;
      } else if( pattern[0] == 'H' || pattern[0] == 'h' ) {
	setExecutionType(desc.nrec, "HYB");
	pattern++;
	if( desc.nproc_summa == 6 )
	  strassenHybrid6( A, B, C, desc, work );
	else if( desc.nproc_summa == 5 )
	  strassenHybrid5( A, B, C, desc, work );
	else if( desc.nproc_summa == 4 )
	  strassenHybrid4( A, B, C, desc, work );
	else if( desc.nproc_summa == 3 )
	  strassenHybrid3( A, B, C, desc, work );
	else if( desc.nproc_summa == 2 )
	  strassenHybrid2( A, B, C, desc, work );
	else if( desc.nproc_summa == 1 )
	  strassenDFS( A, B, C, desc, work );
	else {
	  printf("Hybrid not supported on %d procesors\n", desc.nproc_summa);
	  MPI_Finalize();
	  exit(-1);
	}
	pattern--;
      } 
      else {
	printf("Invalid pattern\n");
	MPI_Finalize();
	exit(-1);
      }
    }

  }
}

void addMatrices( int numEntries, double *C, double *A, double *B ) {
  increaseAdditions(numEntries);
  startTimer(TIMER_ADD);
#pragma omp parallel for schedule(static, (numEntries+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < numEntries; i++ )
    C[i] = A[i] + B[i];
  stopTimer(TIMER_ADD);
}
void subMatrices( int numEntries, double *C, double *A, double *B ) {
  increaseAdditions(numEntries);
  startTimer(TIMER_ADD);
#pragma omp parallel for schedule(static, (numEntries+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < numEntries; i++ )
    C[i] = A[i] - B[i];
  stopTimer(TIMER_ADD);
}
// useful to improve cache behavior if there is some overlap.  It is safe for T_i to be the same as S_j* as long as i<j.  That is, operations will happen in the order specified
void tripleSubMatrices(int numEntries, double *T1, double *S11, double *S12, double *T2,
		       double *S21, double *S22, double *T3, double *S31, double *S32) {
  increaseAdditions(3*numEntries);
  startTimer(TIMER_ADD);
#pragma omp parallel for schedule(static, (numEntries+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < numEntries; i++ ) {
      T1[i] = S11[i] - S12[i];
      T2[i] = S21[i] - S22[i];
      T3[i] = S31[i] - S32[i];
  }
  stopTimer(TIMER_ADD);
}

void tripleAddMatrices(int numEntries, double *T1, double *S11, double *S12, double *T2,
		       double *S21, double *S22, double *T3, double *S31, double *S32) {
  increaseAdditions(3*numEntries);
  startTimer(TIMER_ADD);
#pragma omp parallel for schedule(static, (numEntries+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < numEntries; i++ ) {
      T1[i] = S11[i] + S12[i];
      T2[i] = S21[i] + S22[i];
      T3[i] = S31[i] + S32[i];
  }
  stopTimer(TIMER_ADD);
}

void addSubMatrices(int numEntries, double *T1, double *S11, double *S12, double *T2,
		       double *S21, double *S22 ) {
  increaseAdditions(2*numEntries);
  startTimer(TIMER_ADD);
#pragma omp parallel for schedule(static, (numEntries+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < numEntries; i++ ) {
      T1[i] = S11[i] + S12[i];
      T2[i] = S21[i] - S22[i];
  }
  stopTimer(TIMER_ADD);
}

void addSubSubSubMatrices(int numEntries, double *T1, double *S11, double *S12, double *T2,
			  double *S21, double *S22, double *T3, double *S31, double *S32, 
			  double* T4, double *S41, double *S42 ) {
  increaseAdditions(4*numEntries);
  startTimer(TIMER_ADD);
#pragma omp parallel for schedule(static, (numEntries+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < numEntries; i++ ) {
      T1[i] = S11[i] + S12[i];
      T2[i] = S21[i] - S22[i];
      T3[i] = S31[i] - S32[i];
      T4[i] = S41[i] - S42[i];
  }
  stopTimer(TIMER_ADD);
}

void quadrupleSubMatrices(int numEntries, double *T1, double *S11, double *S12, double *T2,
			  double *S21, double *S22, double *T3, double *S31, double *S32, 
			  double* T4, double *S41, double *S42 ) {
  increaseAdditions(4*numEntries);
  startTimer(TIMER_ADD);
#pragma omp parallel for schedule(static, (numEntries+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < numEntries; i++ ) {
      T1[i] = S11[i] - S12[i];
      T2[i] = S21[i] - S22[i];
      T3[i] = S31[i] - S32[i];
      T4[i] = S41[i] - S42[i];
  }
  stopTimer(TIMER_ADD);
}

void calcCij( int numEntries, double *C11, double *C12, double *C21, double *C22, double *Q1, double *Q2, double *Q3, double *Q4, double *Q5, double *Q6, double *Q7) {
  double U1,U2,U3;
  increaseAdditions(7*numEntries);
  startTimer(TIMER_ADD);
#pragma omp parallel for schedule(static, numEntries/NUM_THREADS) private(U1,U2,U3)
  for( int j = 0; j < numEntries; j++ ) {
    C11[j] = Q1[j] + Q2[j];
    U1 = Q1[j] + Q4[j];
    U2 = U1 + Q5[j];
    C21[j] = U2 - Q7[j];
    C22[j] = U2 + Q3[j];
    U3 = U1 + Q3[j];
    C12[j] = U3 + Q6[j];
  }
  stopTimer(TIMER_ADD);
}

void strassenBFS( double *A, double *B, double *C, MatDescriptor desc, double *workPassThrough ) {
#ifdef SANITY_CHECKS
  verifyDescriptor( desc );
#endif
  // halfDesc will be for the submatrices, such as A11, and the linear combinations of them, the T_.
  MatDescriptor halfDesc = desc;
  halfDesc.lda /= 2;
  halfDesc.nrec -= 1;
#ifdef SANITY_CHECKS
  verifyDescriptor( halfDesc );
#endif

  int numEntriesHalf = numEntriesPerProc(halfDesc);
  double *A11 = A;
  double *A21 = A+numEntriesHalf;
  double *A12 = A+2*numEntriesHalf;
  double *A22 = A+3*numEntriesHalf;
  double *B11 = B;
  double *B21 = B+numEntriesHalf;
  double *B12 = B+2*numEntriesHalf;
  double *B22 = B+3*numEntriesHalf;
  double *C11 = C;
  double *C21 = C+numEntriesHalf;
  double *C12 = C+2*numEntriesHalf;
  double *C22 = C+3*numEntriesHalf;

  // these will be the left and right factors of the product we will compute
  MatDescriptor newDesc = halfDesc;
  newDesc.nproc /= 7;
  if( newDesc.nprocr > newDesc.nprocc ) {
    newDesc.nprocr /= 7;
  } 
 else {
    newDesc.nprocc /= 7;
  }
  int numEntriesNew = numEntriesPerProc(newDesc);
#ifdef SANITY_CHECKS
  assert( 7*numEntriesHalf == numEntriesNew );
#endif
  // this will hold, in sequence, each of the T's as it is calculated, before it is redistributed.

  double *LF = allocate( numEntriesNew );
  double *RF = allocate( numEntriesNew );
  double *work1 = allocate( numEntriesNew );
  double *work2 = allocate( numEntriesNew );

  double *T3 = LF;
  double *T4 = LF+numEntriesHalf;
  double *T5 = LF+2*numEntriesHalf;
  double *T6 = LF+3*numEntriesHalf;
  double *S3 = RF;
  double *S4 = RF+numEntriesHalf;
  double *S5 = RF+2*numEntriesHalf;
  double *S7 = RF+3*numEntriesHalf;
  int TPos[] = {0,1,6};
  MPI_Request *reqs1, *reqs2;
  gatherMatrices1( halfDesc, A11, A12, A22, TPos, newDesc, work1, &reqs1 );
  int SPos[] = {0,1,5};
  gatherMatrices1( halfDesc, B11, B21, B22, SPos, newDesc, work2, &reqs2 );
  // calculate T3, T4, T5, T6 (T1=A11, T2=A12, T7=A22 don't need computation
  addSubSubSubMatrices( numEntriesHalf, T3, A21, A22, T4, T3, A11, T5, A11, A21, T6, A12, T4 );
  int TPos2[] = {2,3,4,5};
  gatherMatrices2( halfDesc, T3, T4, T5, T6, TPos2, newDesc, work1, reqs1 );

  // calculate S3, S4, S5, S7 (S1=B11, S2=B21, S6=B22 don't need computation)
  quadrupleSubMatrices(numEntriesHalf, S5, B22, B12, S3, B12, B11, S4, B22, S3, S7, S4, B21);
  int SPos2[] = {2,3,4,6};
  gatherMatrices2( halfDesc, S3, S4, S5, S7, SPos2, newDesc, work2, reqs2 );

  // first time we need LF, we won't need the T's by the time LF is written
  gatherMatrices3( halfDesc, newDesc, work1, reqs1, LF );
  gatherMatrices3( halfDesc, newDesc, work2, reqs2, RF );

  // done with T1234567, now need space for the product
  double *P = work2;

  // recursive multiply call
  multiplyInternal( LF, RF, P, newDesc, workPassThrough );

  // done with RF, use the space for the Q's
  double *Q1234567 = RF;
  double *Q1 = RF;
  double *Q2 = RF + numEntriesHalf;
  double *Q3 = RF + 2*numEntriesHalf;
  double *Q4 = RF + 3*numEntriesHalf;
  double *Q5 = RF + 4*numEntriesHalf;
  double *Q6 = RF + 5*numEntriesHalf;
  double *Q7 = RF + 6*numEntriesHalf;
  scatterMatrices( newDesc, P, halfDesc, Q1234567, work1 );

  deallocate( LF, numEntriesNew );
  deallocate( work1, numEntriesNew );
  deallocate( work2, numEntriesNew );

  // final calculations of the Q's into C.
  calcCij( numEntriesHalf, C11, C12, C21, C22, Q1, Q2, Q3, Q4, Q5, Q6, Q7 );

  deallocate( RF, numEntriesNew );
}


void strassenDFS( double *A, double *B, double *C, MatDescriptor desc, double *workPassThrough ) {
#ifdef SANITY_CHECKS
  verifyDescriptor( desc );
#endif
  MatDescriptor halfDesc = desc;
  halfDesc.lda /= 2;
  halfDesc.nrec -= 1;
#ifdef SANITY_CHECKS
  verifyDescriptor( halfDesc );
#endif
  // submatrices; these are described by halfDesc
  int numEntriesHalf = numEntriesPerProc(halfDesc);
  double *A11 = A;
  double *A21 = A+numEntriesHalf;
  double *A12 = A+2*numEntriesHalf;
  double *A22 = A+3*numEntriesHalf;
  double *B11 = B;
  double *B21 = B+numEntriesHalf;
  double *B12 = B+2*numEntriesHalf;
  double *B22 = B+3*numEntriesHalf;
  double *C11 = C;
  double *C21 = C+numEntriesHalf;
  double *C12 = C+2*numEntriesHalf;
  double *C22 = C+3*numEntriesHalf;

#ifdef DAXPY
  double done = 1.;
  double dnone = -1.;
  int ione = 1;
  // six registers.  halfDesc is the descriptor for these
  double *R1 = C21;
  double *R2 = allocate( numEntriesHalf );
  double *R3 = C11; // fixed
  double *R4 = C22; // fixed
  double *R6 = allocate( numEntriesHalf );
  double *R5 = C12; // fixed

  double *S5 = R1;
  double *S3 = R2;
  double *S4 = R3;
  startTimer(TIMER_ADD);
  dcopy_(&numEntriesHalf, B22, &ione, S5, &ione );
  daxpy_(&numEntriesHalf, &dnone, B12, &ione, S5, &ione );
  dcopy_(&numEntriesHalf, B12, &ione, S3, &ione );
  daxpy_(&numEntriesHalf, &dnone, B11, &ione, S3, &ione );
  dcopy_(&numEntriesHalf, B22, &ione, S4, &ione );
  daxpy_(&numEntriesHalf, &dnone, S3, &ione, S4, &ione );
  double *T5 = R4;
  double *T3 = R6;
  dcopy_(&numEntriesHalf, A21, &ione, T3, &ione );
  daxpy_(&numEntriesHalf, &done, A22, &ione, T3, &ione );
  dcopy_(&numEntriesHalf, A11, &ione, T5, &ione );
  daxpy_(&numEntriesHalf, &dnone, A21, &ione, T5, &ione );
  stopTimer(TIMER_ADD);
  double *Q5 = R5;
  multiplyInternal( T5, S5, Q5, halfDesc, workPassThrough); 
  double *Q3 = R4;// =C22
  multiplyInternal( T3, S3, Q3, halfDesc, workPassThrough);
  double *T4 = R6; // =T3
  startTimer(TIMER_ADD);
  daxpy_(&numEntriesHalf, &dnone, A11, &ione, T4, &ione );
  stopTimer(TIMER_ADD);
  double *Q4 = R2;
  multiplyInternal( T4, S4, Q4, halfDesc, workPassThrough);
  double *T6 = R6; // =T4
  startTimer(TIMER_ADD);
  daxpy_(&numEntriesHalf, &dnone, A12, &ione, T6, &ione);
  dscal_(&numEntriesHalf, &dnone, T6, &ione);
  double *S7 = R3; // =S4
  daxpy_(&numEntriesHalf, &dnone, B21, &ione, S7, &ione);
  stopTimer(TIMER_ADD);
  double *Q7 = R1;
  multiplyInternal( A22, S7, Q7, halfDesc, workPassThrough);
  double *Q1 = R3;// = C11
  multiplyInternal( A11, B11, Q1, halfDesc, workPassThrough);
  double *U1 = R2;// =Q4
  double *U2 = R5;// =Q5
  double *U3 = R2;// =U1
  startTimer(TIMER_ADD);
  daxpy_(&numEntriesHalf, &done, Q1, &ione, U1, &ione);
  daxpy_(&numEntriesHalf, &done, U1, &ione, U2, &ione);
  daxpy_(&numEntriesHalf, &done, Q3, &ione, U3, &ione);
  daxpy_(&numEntriesHalf, &done, U2, &ione, C22, &ione);
  daxpy_(&numEntriesHalf, &dnone, U2, &ione, C21, &ione); 
  dscal_(&numEntriesHalf, &dnone, C21, &ione);
  stopTimer(TIMER_ADD);  
  double *Q2 = R5;
  multiplyInternal( A12, B21, Q2, halfDesc, workPassThrough);
  daxpy_(&numEntriesHalf, &done, Q2, &ione, Q1, &ione);
  double *Q6 = R5; // =C12
  multiplyInternal(T6, B22, Q6, halfDesc, workPassThrough);
  startTimer(TIMER_ADD);
  daxpy_(&numEntriesHalf, &done, U3, &ione, Q6, &ione);
  stopTimer(TIMER_ADD);  
  deallocate(R6, numEntriesHalf);
  deallocate(R2, numEntriesHalf);
#else
  // six registers.  halfDesc is the descriptor for these
  double *R1 = C21;
  double *R2 = allocate( numEntriesHalf );
  double *R3 = C11;
  double *R4 = C22;
  double *R5 = allocate( numEntriesHalf );
  double *R6 = C12;

  double *S5 = R1;
  double *S3 = R2;
  double *S4 = R3;
  tripleSubMatrices(numEntriesHalf, S5, B22, B12, S3, B12, B11, S4, B22, S3);
  double *T5 = R4;
  double *T3 = R6; // was R1
  addSubMatrices(numEntriesHalf, T3, A21, A22, T5, A11, A21);
  double *Q5 = R5;
  multiplyInternal( T5, S5, Q5, halfDesc, workPassThrough); 
  double *Q3 = R4;
  multiplyInternal( T3, S3, Q3, halfDesc, workPassThrough);
  double *T4 = R6;
  subMatrices(numEntriesHalf, T4, T3, A11);
  double *Q4 = R2;
  multiplyInternal( T4, S4, Q4, halfDesc, workPassThrough);
  double *T6 = R6;
  subMatrices(numEntriesHalf, T6, A12, T4);
  double *S7 = R3;
  subMatrices(numEntriesHalf, S7, S4, B21);
  double *Q7 = R1;
  multiplyInternal( A22, S7, Q7, halfDesc, workPassThrough);
  double *Q1 = R3;
  multiplyInternal( A11, B11, Q1, halfDesc, workPassThrough);
  double *U1 = R2;
  double *U2 = R5;
  double *U3 = R2;
  tripleAddMatrices(numEntriesHalf, U1, Q1, Q4, U2, U1, Q5, U3, U1, Q3);
  addSubMatrices(numEntriesHalf, C22, U2, Q3, C21, U2, Q7);
  double *Q2 = R5;
  multiplyInternal( A12, B21, Q2, halfDesc, workPassThrough);
  addMatrices(numEntriesHalf, C11, Q1, Q2);
  double *Q6 = R5;
  multiplyInternal(T6, B22, Q6, halfDesc, workPassThrough);
  addMatrices(numEntriesHalf, C12, U3, Q6);
  deallocate(R5, numEntriesHalf);
  deallocate(R2, numEntriesHalf);
#endif
}

// do a distributed hybrid with 6 processors.  Only works if the matrices are currently in N-morton-column-block, so need nproc=1.  But we wouldn't want to do this before nproc=1 anyway.
void strassenHybrid6( double *A, double *B, double *C, MatDescriptor desc, double *workPassThrough ) {
#ifdef SANITY_CHECKS
  verifyDescriptor( desc );
  assert( desc.nproc == 1 );
  assert( desc.nrec >= 1 );
  assert( desc.nproc_summa == 6 );
#endif
  MatDescriptor halfDesc = desc;
  halfDesc.lda /= 2;
  halfDesc.nrec -= 1;
  MatDescriptor newDesc = halfDesc;
  newDesc.nproc_summa = 1;
  int numEntriesHalf = numEntriesPerProc(halfDesc);
  int numEntriesNew = numEntriesPerProc(newDesc);
#ifdef SANITY_CHECKS
  verifyDescriptor( halfDesc );
  verifyDescriptor( newDesc );
  assert( 6*numEntriesHalf == numEntriesNew );
#endif

  double *A11 = A;
  double *A21 = A+numEntriesHalf;
  double *A12 = A+2*numEntriesHalf;
  double *A22 = A+3*numEntriesHalf;
  double *B11 = B;
  double *B21 = B+numEntriesHalf;
  double *B12 = B+2*numEntriesHalf;
  double *B22 = B+3*numEntriesHalf;
  double *C11 = C;
  double *C21 = C+numEntriesHalf;
  double *C12 = C+2*numEntriesHalf;
  double *C22 = C+3*numEntriesHalf;


  double *LF = allocate( numEntriesNew );
  double *RF = allocate( numEntriesNew );
  double *work1 = allocate( numEntriesNew );
  double *work2 = allocate( numEntriesNew );
  double *T3 = LF;
  double *T4 = LF+numEntriesHalf;
  double *T6 = LF+2*numEntriesHalf;
  double *S3 = RF;
  double *S4 = RF+numEntriesHalf;
  double *S7 = RF+2*numEntriesHalf;
  double *T5 = allocate( numEntriesHalf );
  double *S5 = allocate( numEntriesHalf );
  // we will distribute: T1 -> 0, T2 -> 1, T3, -> 2, T4 -> 3, T6 -> 4, T7 -> 5.  T5 will remain undistributed

  MPI_Request *reqs1, *reqs2;
  gatherMatricesS0( halfDesc, newDesc, &reqs1 );
  double *Tsrc[] = {A11, A12, A22};
  int TPos[] = {0,1,5};
  gatherMatricesS1( halfDesc, Tsrc, TPos, 3, newDesc, work1, reqs1 );
  gatherMatricesS0( halfDesc, newDesc, &reqs2 );
  double *Ssrc[] = {B11, B21, B22};
  int SPos[] = {0,1,4};
  gatherMatricesS1( halfDesc, Ssrc, SPos, 3, newDesc, work2, reqs2 );
  // calculate T3, T4, T5, T6 (T1=A11, T2=A12, T7=A22 don't need computation
  addSubSubSubMatrices( numEntriesHalf, T3, A21, A22, T4, T3, A11, T5, A11, A21, T6, A12, T4 );
  double *Tsrc2[] = {T3, T4, T6};
  int TPos2[] = {2,3,4};
  gatherMatricesS1( halfDesc, Tsrc2, TPos2, 3, newDesc, work1, reqs1 );

  // calculate S3, S4, S5, S7 (S1=B11, S2=B21, S6=B22 don't need computation)
  quadrupleSubMatrices(numEntriesHalf, S5, B22, B12, S3, B12, B11, S4, B22, S3, S7, S4, B21);
  double *Ssrc2[] = {S3, S4, S7};
  int SPos2[] = {2,3,5};
  gatherMatricesS1( halfDesc, Ssrc2, SPos2, 3, newDesc, work2, reqs2 );

  // first time we need LF, we won't need the T's by the time LF is written
  gatherMatricesS2( reqs1, halfDesc, newDesc, work1, LF );
  gatherMatricesS2( reqs2, halfDesc, newDesc, work2, RF );

  // done with work1, use space for product
  double *P = work1;


  // recursive multiply call
  multiplyInternal( LF, RF, P, newDesc, workPassThrough );
  // start scattering the products.  Done with LF,RF
  double *Q1 = RF;
  double *Q2 = RF+numEntriesHalf;
  double *Q3 = RF+2*numEntriesHalf;
  double *Q4 = RF+3*numEntriesHalf;
  double *Q6 = RF+4*numEntriesHalf;
  double *Q7 = RF+5*numEntriesHalf;
  double *Q5 = LF;
  int QPos[] = { 0, 1, 2, 3, 4, 5 };
  double *Qsrc[] = { Q1, Q2, Q3, Q4, Q6, Q7 };
  scatterMatricesS0( halfDesc, newDesc, &reqs1, P, work2 );
  scatterMatricesS1( halfDesc, Qsrc, QPos, 6, newDesc, work2, reqs1 );
  // and the last one
  multiplyInternal( T5, S5, Q5, halfDesc, workPassThrough );

  // finish scattering the products.  Same function works either way
  scatterMatricesS2( halfDesc, reqs1 );

  // final calculations of the Q's into C.
  calcCij( numEntriesHalf, C11, C12, C21, C22, Q1, Q2, Q3, Q4, Q5, Q6, Q7 );
  
  deallocate( LF, numEntriesNew );
  deallocate( RF, numEntriesNew );
  deallocate( work1, numEntriesNew );
  deallocate( work2, numEntriesNew );
  deallocate( T5, numEntriesHalf );
  deallocate( S5, numEntriesHalf );
}

// do a distributed hybrid with 5 processors.  
void strassenHybrid5( double *A, double *B, double *C, MatDescriptor desc, double *workPassThrough ) {
#ifdef SANITY_CHECKS
  verifyDescriptor( desc );
  assert( desc.nproc == 1 );
  assert( desc.nrec >= 1 );
  assert( desc.nproc_summa == 5 );
#endif
  MatDescriptor halfDesc = desc;
  halfDesc.lda /= 2;
  halfDesc.nrec -= 1;
  MatDescriptor newDesc = halfDesc;
  newDesc.nproc_summa = 1;
  int numEntriesHalf = numEntriesPerProc(halfDesc);
  int numEntriesNew = numEntriesPerProc(newDesc);
#ifdef SANITY_CHECKS
  verifyDescriptor( halfDesc );
  verifyDescriptor( newDesc );
  assert( 5*numEntriesHalf == numEntriesNew );
#endif

  double *A11 = A;
  double *A21 = A+numEntriesHalf;
  double *A12 = A+2*numEntriesHalf;
  double *A22 = A+3*numEntriesHalf;
  double *B11 = B;
  double *B21 = B+numEntriesHalf;
  double *B12 = B+2*numEntriesHalf;
  double *B22 = B+3*numEntriesHalf;
  double *C11 = C;
  double *C21 = C+numEntriesHalf;
  double *C12 = C+2*numEntriesHalf;
  double *C22 = C+3*numEntriesHalf;


  double *LF = allocate( numEntriesNew );
  double *RF = allocate( numEntriesNew );
  double *work1 = allocate( numEntriesNew );
  double *work2 = allocate( numEntriesNew );
  double *T3 = LF;
  double *T6 = LF+numEntriesHalf;
  double *S3 = RF;
  double *S7 = RF+numEntriesHalf;
  double *T4 = allocate( numEntriesHalf );
  double *S4 = allocate( numEntriesHalf );
  double *T5 = allocate( numEntriesHalf );
  double *S5 = allocate( numEntriesHalf );
  // we will distribute: T1 -> 0, T2 -> 1, T3, -> 2, T6 -> 3, T7 -> 4.  T4,T5 will remain undistributed

  MPI_Request *reqs1, *reqs2;
  gatherMatricesS0( halfDesc, newDesc, &reqs1 );
  double *Tsrc[] = {A11, A12, A22}; // which are T1,T2,T7
  int TPos[] = {0,1,4};
  gatherMatricesS1( halfDesc, Tsrc, TPos, 3, newDesc, work1, reqs1 );
  gatherMatricesS0( halfDesc, newDesc, &reqs2 );
  double *Ssrc[] = {B11, B21, B22}; // which are S1,S2,S6
  int SPos[] = {0,1,3};
  gatherMatricesS1( halfDesc, Ssrc, SPos, 3, newDesc, work2, reqs2 );
  // calculate T3, T4, T5, T6 (T1=A11, T2=A12, T7=A22 don't need computation
  addSubSubSubMatrices( numEntriesHalf, T3, A21, A22, T4, T3, A11, T5, A11, A21, T6, A12, T4 );
  double *Tsrc2[] = {T3, T6};
  int TPos2[] = {2,3};
  gatherMatricesS1( halfDesc, Tsrc2, TPos2, 2, newDesc, work1, reqs1 );

  // calculate S3, S4, S5, S7 (S1=B11, S2=B21, S6=B22 don't need computation)
  quadrupleSubMatrices(numEntriesHalf, S5, B22, B12, S3, B12, B11, S4, B22, S3, S7, S4, B21);
  double *Ssrc2[] = {S3, S7};
  int SPos2[] = {2,4};
  gatherMatricesS1( halfDesc, Ssrc2, SPos2, 2, newDesc, work2, reqs2 );

  // first time we need LF, we won't need the T's by the time LF is written
  gatherMatricesS2( reqs1, halfDesc, newDesc, work1, LF );
  gatherMatricesS2( reqs2, halfDesc, newDesc, work2, RF );

  // done with work1, use space for product
  double *P = work1;


  // recursive multiply call
  multiplyInternal( LF, RF, P, newDesc, workPassThrough );
  // start scattering the products.  Done with LF,RF
  double *Q1 = RF;
  double *Q2 = RF+numEntriesHalf;
  double *Q3 = RF+2*numEntriesHalf;
  double *Q6 = RF+3*numEntriesHalf;
  double *Q7 = RF+4*numEntriesHalf;
  double *Q4 = LF;
  double *Q5 = LF+numEntriesHalf;
  int QPos[] = { 0, 1, 2, 3, 4 };
  double *Qsrc[] = { Q1, Q2, Q3, Q6, Q7 };
  scatterMatricesS0( halfDesc, newDesc, &reqs1, P, work2 );
  scatterMatricesS1( halfDesc, Qsrc, QPos, 5, newDesc, work2, reqs1 );
  // and the last two
  multiplyInternal( T4, S4, Q4, halfDesc, workPassThrough );
  multiplyInternal( T5, S5, Q5, halfDesc, workPassThrough );

  // finish scattering the products.  Same function works either way
  scatterMatricesS2( halfDesc, reqs1 );

  // final calculations of the Q's into C.
  calcCij( numEntriesHalf, C11, C12, C21, C22, Q1, Q2, Q3, Q4, Q5, Q6, Q7 );
  
  deallocate( LF, numEntriesNew );
  deallocate( RF, numEntriesNew );
  deallocate( work1, numEntriesNew );
  deallocate( work2, numEntriesNew );
  deallocate( T5, numEntriesHalf );
  deallocate( S5, numEntriesHalf );
  deallocate( T4, numEntriesHalf );
  deallocate( S4, numEntriesHalf );
}

// do a distributed hybrid with 4 processors.  Currently does the last 3 all together, but that could be improved
void strassenHybrid4( double *A, double *B, double *C, MatDescriptor desc, double *workPassThrough ) {
#ifdef SANITY_CHECKS
  verifyDescriptor( desc );
  assert( desc.nproc == 1 );
  assert( desc.nrec >= 1 );
  assert( desc.nproc_summa == 4 );
#endif
  MatDescriptor halfDesc = desc;
  halfDesc.lda /= 2;
  halfDesc.nrec -= 1;
  MatDescriptor newDesc = halfDesc;
  newDesc.nproc_summa = 1;
  int numEntriesHalf = numEntriesPerProc(halfDesc);
  int numEntriesNew = numEntriesPerProc(newDesc);
#ifdef SANITY_CHECKS
  verifyDescriptor( halfDesc );
  verifyDescriptor( newDesc );
  assert( 4*numEntriesHalf == numEntriesNew );
#endif

  double *A11 = A;
  double *A21 = A+numEntriesHalf;
  double *A12 = A+2*numEntriesHalf;
  double *A22 = A+3*numEntriesHalf;
  double *B11 = B;
  double *B21 = B+numEntriesHalf;
  double *B12 = B+2*numEntriesHalf;
  double *B22 = B+3*numEntriesHalf;
  double *C11 = C;
  double *C21 = C+numEntriesHalf;
  double *C12 = C+2*numEntriesHalf;
  double *C22 = C+3*numEntriesHalf;


  double *LF = allocate( numEntriesNew );
  double *RF = allocate( numEntriesNew );
  double *work1 = allocate( numEntriesNew );
  double *work2 = allocate( numEntriesNew );
  double *T6 = LF;
  double *S7 = RF;
  double *T3 = allocate( numEntriesHalf );
  double *S3 = allocate( numEntriesHalf );
  double *T4 = allocate( numEntriesHalf );
  double *S4 = allocate( numEntriesHalf );
  double *T5 = allocate( numEntriesHalf );
  double *S5 = allocate( numEntriesHalf );
  // we will distribute: T1 -> 0, T2 -> 1, T6 -> 2, T7 -> 3.  T3,T4,T5 will remain undistributed

  MPI_Request *reqs1, *reqs2;
  gatherMatricesS0( halfDesc, newDesc, &reqs1 );
  double *Tsrc[] = {A11, A12, A22}; // which are T1,T2,T7
  int TPos[] = {0,1,3};
  gatherMatricesS1( halfDesc, Tsrc, TPos, 3, newDesc, work1, reqs1 );
  gatherMatricesS0( halfDesc, newDesc, &reqs2 );
  double *Ssrc[] = {B11, B21, B22}; // which are S1,S2,S6
  int SPos[] = {0,1,2};
  gatherMatricesS1( halfDesc, Ssrc, SPos, 3, newDesc, work2, reqs2 );
  // calculate T3, T4, T5, T6 (T1=A11, T2=A12, T7=A22 don't need computation
  addSubSubSubMatrices( numEntriesHalf, T3, A21, A22, T4, T3, A11, T5, A11, A21, T6, A12, T4 );
  double *Tsrc2[] = {T6};
  int TPos2[] = {2};
  gatherMatricesS1( halfDesc, Tsrc2, TPos2, 1, newDesc, work1, reqs1 );

  // calculate S3, S4, S5, S7 (S1=B11, S2=B21, S6=B22 don't need computation)
  quadrupleSubMatrices(numEntriesHalf, S5, B22, B12, S3, B12, B11, S4, B22, S3, S7, S4, B21);
  double *Ssrc2[] = {S7};
  int SPos2[] = {3};
  gatherMatricesS1( halfDesc, Ssrc2, SPos2, 1, newDesc, work2, reqs2 );

  // first time we need LF, we won't need the T's by the time LF is written
  gatherMatricesS2( reqs1, halfDesc, newDesc, work1, LF );
  gatherMatricesS2( reqs2, halfDesc, newDesc, work2, RF );

  // done with work1, use space for product
  double *P = work1;


  // recursive multiply call
  multiplyInternal( LF, RF, P, newDesc, workPassThrough );
  // start scattering the products.  Done with LF,RF
  double *Q1 = RF;
  double *Q2 = RF+numEntriesHalf;
  double *Q6 = RF+2*numEntriesHalf;
  double *Q7 = RF+3*numEntriesHalf;
  double *Q4 = LF;
  double *Q5 = LF+numEntriesHalf;
  double *Q3 = LF+2*numEntriesHalf;
  int QPos[] = { 0, 1, 2, 3 };
  double *Qsrc[] = { Q1, Q2, Q6, Q7 };
  scatterMatricesS0( halfDesc, newDesc, &reqs1, P, work2 );
  scatterMatricesS1( halfDesc, Qsrc, QPos, 4, newDesc, work2, reqs1 );
  // and the last three
  multiplyInternal( T3, S3, Q3, halfDesc, workPassThrough );
  multiplyInternal( T4, S4, Q4, halfDesc, workPassThrough );
  multiplyInternal( T5, S5, Q5, halfDesc, workPassThrough );

  // finish scattering the products.
  scatterMatricesS2( halfDesc, reqs1 );

  // final calculations of the Q's into C.
  calcCij( numEntriesHalf, C11, C12, C21, C22, Q1, Q2, Q3, Q4, Q5, Q6, Q7 );
  
  deallocate( LF, numEntriesNew );
  deallocate( RF, numEntriesNew );
  deallocate( work1, numEntriesNew );
  deallocate( work2, numEntriesNew );
  deallocate( T5, numEntriesHalf );
  deallocate( S5, numEntriesHalf );
  deallocate( T4, numEntriesHalf );
  deallocate( S4, numEntriesHalf );
  deallocate( T3, numEntriesHalf );
  deallocate( S3, numEntriesHalf );
}

// do a distributed hybrid with 3 processors.
void strassenHybrid3( double *A, double *B, double *C, MatDescriptor desc, double *workPassThrough ) {
#ifdef SANITY_CHECKS
  verifyDescriptor( desc );
  assert( desc.nproc == 1 );
  assert( desc.nrec >= 1 );
  assert( desc.nproc_summa == 3 );
#endif
  MatDescriptor halfDesc = desc;
  halfDesc.lda /= 2;
  halfDesc.nrec -= 1;
  MatDescriptor newDesc = halfDesc;
  newDesc.nproc_summa = 1;
  int numEntriesHalf = numEntriesPerProc(halfDesc);
  int numEntriesNew = numEntriesPerProc(newDesc);
#ifdef SANITY_CHECKS
  verifyDescriptor( halfDesc );
  verifyDescriptor( newDesc );
  assert( 3*numEntriesHalf == numEntriesNew );
#endif

  double *A11 = A;
  double *A21 = A+numEntriesHalf;
  double *A12 = A+2*numEntriesHalf;
  double *A22 = A+3*numEntriesHalf;
  double *B11 = B;
  double *B21 = B+numEntriesHalf;
  double *B12 = B+2*numEntriesHalf;
  double *B22 = B+3*numEntriesHalf;
  double *C11 = C;
  double *C21 = C+numEntriesHalf;
  double *C12 = C+2*numEntriesHalf;
  double *C22 = C+3*numEntriesHalf;

  double *LF = allocate( numEntriesNew );
  double *RF = allocate( numEntriesNew );
  double *work1 = allocate( numEntriesNew );
  double *work2 = allocate( numEntriesNew );
  double *P = allocate( numEntriesNew );
  double *S7 = RF;
  double *T3 = allocate( numEntriesHalf );
  double *S3 = allocate( numEntriesHalf );
  double *T4 = allocate( numEntriesHalf );
  double *S4 = allocate( numEntriesHalf );
  double *T5 = allocate( numEntriesHalf );
  double *S5 = allocate( numEntriesHalf );
  double *T6 = allocate( numEntriesHalf );
  double *Q1 = allocate( numEntriesHalf );
  double *Q2 = allocate( numEntriesHalf );
  double *Q3 = allocate( numEntriesHalf );
  double *Q4 = allocate( numEntriesHalf );
  double *Q5 = allocate( numEntriesHalf );
  double *Q6 = allocate( numEntriesHalf );
  double *Q7 = allocate( numEntriesHalf );
  double *T1 = A11;
  double *T2 = A12;
  double *T7 = A22;
  double *S1 = B11;
  double *S2 = B21;
  double *S6 = B22;
  // we will distribute: T1 -> 0, T2 -> 1, T6 -> 2; then T3 ->0, T4 ->1, T7 -> 2; T5 will remain undistributed

  MPI_Request *reqs1, *reqs2;
  gatherMatricesS0( halfDesc, newDesc, &reqs1 );
  double *Tsrc[] = {T1, T2};
  int TPos[] = {0,1};
  gatherMatricesS1( halfDesc, Tsrc, TPos, 2, newDesc, work1, reqs1 );
  gatherMatricesS0( halfDesc, newDesc, &reqs2 );
  double *Ssrc[] = {S1, S2, S6};
  int SPos[] = {0,1,2};
  gatherMatricesS1( halfDesc, Ssrc, SPos, 3, newDesc, work2, reqs2 );
  // calculate T3, T4, T5, T6 (T1=A11, T2=A12, T7=A22 don't need computation
  addSubSubSubMatrices( numEntriesHalf, T3, A21, A22, T4, T3, A11, T5, A11, A21, T6, A12, T4 );
  double *Tsrc2[] = {T6};
  int TPos2[] = {2};
  gatherMatricesS1( halfDesc, Tsrc2, TPos2, 1, newDesc, work1, reqs1 );
  // first time we need LF, we won't need the T's by the time LF is written
  gatherMatricesS2( reqs1, halfDesc, newDesc, work1, LF );
  gatherMatricesS2( reqs2, halfDesc, newDesc, work2, RF );

  // recursive multiply call
  multiplyInternal( LF, RF, P, newDesc, workPassThrough );

  int QPos[] = { 0, 1, 2 };
  double *Qsrc[] = { Q1, Q2, Q6 };
  scatterMatricesS0( halfDesc, newDesc, &reqs1, P, work2 );
  scatterMatricesS1( halfDesc, Qsrc, QPos, 3, newDesc, work2, reqs1 );
  quadrupleSubMatrices(numEntriesHalf, S5, B22, B12, S3, B12, B11, S4, B22, S3, S7, S4, B21);
  scatterMatricesS2( halfDesc, reqs1 );

  // now do the second round of gather / multiply / scatter T3 ->0, T4 ->1, T7 -> 2
  double *Tsrc3[] = {T3,T4,T7};
  gatherMatricesS0( halfDesc, newDesc, &reqs1 );
  gatherMatricesS1( halfDesc, Tsrc3, QPos, 3, newDesc, work1, reqs1 );
  gatherMatricesS0( halfDesc, newDesc, &reqs2 );
  double *Ssrc3[] = {S3,S4,S7};
  gatherMatricesS1( halfDesc, Ssrc3, QPos, 3, newDesc, work2, reqs2 );

  // meanwhile, the last multiply
  multiplyInternal( T5, S5, Q5, halfDesc, workPassThrough );
 
  gatherMatricesS2( reqs1, halfDesc, newDesc, work1, LF );
  gatherMatricesS2( reqs2, halfDesc, newDesc, work2, RF );
  // recursive multiply call
  multiplyInternal( LF, RF, P, newDesc, workPassThrough );

  double *Qsrc2[] = { Q3, Q4, Q7 };
  scatterMatricesS0( halfDesc, newDesc, &reqs1, P, work2 );
  scatterMatricesS1( halfDesc, Qsrc2, QPos, 3, newDesc, work2, reqs1 );
  // nothing useful to do while waiting
  scatterMatricesS2( halfDesc, reqs1 );

  // final calculations of the Q's into C.
  calcCij( numEntriesHalf, C11, C12, C21, C22, Q1, Q2, Q3, Q4, Q5, Q6, Q7 );
  
  deallocate( LF, numEntriesNew );
  deallocate( RF, numEntriesNew );
  deallocate( work1, numEntriesNew );
  deallocate( work2, numEntriesNew );
  deallocate( P, numEntriesNew );
  deallocate( T5, numEntriesHalf );
  deallocate( S5, numEntriesHalf );
  deallocate( T4, numEntriesHalf );
  deallocate( S4, numEntriesHalf );
  deallocate( T3, numEntriesHalf );
  deallocate( S3, numEntriesHalf );
  deallocate( T6, numEntriesHalf );
  deallocate( Q1, numEntriesHalf );
  deallocate( Q2, numEntriesHalf );
  deallocate( Q3, numEntriesHalf );
  deallocate( Q4, numEntriesHalf );
  deallocate( Q5, numEntriesHalf );
  deallocate( Q6, numEntriesHalf );
  deallocate( Q7, numEntriesHalf );
}

// do a distributed hybrid with 2 processors.
void strassenHybrid2( double *A, double *B, double *C, MatDescriptor desc, double *workPassThrough ) {
#ifdef SANITY_CHECKS
  verifyDescriptor( desc );
  assert( desc.nproc == 1 );
  assert( desc.nrec >= 1 );
  assert( desc.nproc_summa == 2 );
#endif
  MatDescriptor halfDesc = desc;
  halfDesc.lda /= 2;
  halfDesc.nrec -= 1;
  MatDescriptor newDesc = halfDesc;
  newDesc.nproc_summa = 1;
  int numEntriesHalf = numEntriesPerProc(halfDesc);
  int numEntriesNew = numEntriesPerProc(newDesc);
#ifdef SANITY_CHECKS
  verifyDescriptor( halfDesc );
  verifyDescriptor( newDesc );
  assert( 2*numEntriesHalf == numEntriesNew );
#endif

  double *A11 = A;
  double *A21 = A+numEntriesHalf;
  double *A12 = A+2*numEntriesHalf;
  double *A22 = A+3*numEntriesHalf;
  double *B11 = B;
  double *B21 = B+numEntriesHalf;
  double *B12 = B+2*numEntriesHalf;
  double *B22 = B+3*numEntriesHalf;
  double *C11 = C;
  double *C21 = C+numEntriesHalf;
  double *C12 = C+2*numEntriesHalf;
  double *C22 = C+3*numEntriesHalf;


  double *LF = allocate( numEntriesNew );
  double *RF = allocate( numEntriesNew );
  double *work1 = allocate( numEntriesNew );
  double *work2 = allocate( numEntriesNew );
  double *work3 = allocate( numEntriesNew );
  double *P = allocate( numEntriesNew );
  double *S7 = allocate( numEntriesHalf );
  double *T3 = allocate( numEntriesHalf );
  double *S3 = allocate( numEntriesHalf );
  double *T4 = allocate( numEntriesHalf );
  double *S4 = allocate( numEntriesHalf );
  double *T5 = allocate( numEntriesHalf );
  double *S5 = allocate( numEntriesHalf );
  double *T6 = allocate( numEntriesHalf );
  double *Q1 = allocate( numEntriesHalf );
  double *Q2 = allocate( numEntriesHalf );
  double *Q3 = allocate( numEntriesHalf );
  double *Q4 = allocate( numEntriesHalf );
  double *Q5 = allocate( numEntriesHalf );
  double *Q6 = allocate( numEntriesHalf );
  double *Q7 = allocate( numEntriesHalf );
  double *T1 = A11;
  double *T2 = A12;
  double *T7 = A22;
  double *S1 = B11;
  double *S2 = B21;
  double *S6 = B22;
  // we will distribute: T1 -> 0, T2 -> 1; then T6 -> 0, T7 -> 1; then T3 ->0, T4 ->1; T5 will remain undistributed

  MPI_Request *reqs1, *reqs2, *reqs3;
  gatherMatricesS0( halfDesc, newDesc, &reqs1 );
  double *Tsrc[] = {T1, T2};
  int TPos[] = {0,1};
  gatherMatricesS1( halfDesc, Tsrc, TPos, 2, newDesc, work1, reqs1 );
  gatherMatricesS0( halfDesc, newDesc, &reqs2 );
  double *Ssrc[] = {S1, S2};
  int SPos[] = {0,1};
  gatherMatricesS1( halfDesc, Ssrc, SPos, 2, newDesc, work2, reqs2 );
  // calculate T3, T4, T5, T6; S3, S4, S5, S7
  addSubSubSubMatrices( numEntriesHalf, T3, A21, A22, T4, T3, A11, T5, A11, A21, T6, A12, T4 );
  quadrupleSubMatrices(numEntriesHalf, S5, B22, B12, S3, B12, B11, S4, B22, S3, S7, S4, B21);
  gatherMatricesS2( reqs1, halfDesc, newDesc, work1, LF );
  gatherMatricesS2( reqs2, halfDesc, newDesc, work2, RF );

  // start the second round of transfers
  gatherMatricesS0( halfDesc, newDesc, &reqs1 );
  double *Tsrc2[] = {T6,T7};
  gatherMatricesS1( halfDesc, Tsrc2, TPos, 2, newDesc, work1, reqs1 );
  gatherMatricesS0( halfDesc, newDesc, &reqs2 );
  double *Ssrc2[] = {S6,S7};
  gatherMatricesS1( halfDesc, Ssrc2, TPos, 2, newDesc, work2, reqs2 );
  // first multiply
  multiplyInternal( LF, RF, P, newDesc, workPassThrough );
  // start transferring out the result
  double *Qsrc[] = { Q1, Q2 };
  scatterMatricesS0( halfDesc, newDesc, &reqs3, P, work3 );
  scatterMatricesS1( halfDesc, Qsrc, TPos, 2, newDesc, work3, reqs3 );
  // finish the second gather
  gatherMatricesS2( reqs1, halfDesc, newDesc, work1, LF );
  gatherMatricesS2( reqs2, halfDesc, newDesc, work2, RF );

  // start the third round of transfers
  gatherMatricesS0( halfDesc, newDesc, &reqs1 );
  double *Tsrc3[] = {T3,T4};
  gatherMatricesS1( halfDesc, Tsrc3, TPos, 2, newDesc, work1, reqs1 );
  gatherMatricesS0( halfDesc, newDesc, &reqs2 );
  double *Ssrc3[] = {S3,S4};
  gatherMatricesS1( halfDesc, Ssrc3, TPos, 2, newDesc, work2, reqs2 );
  // second multiply
  multiplyInternal( LF, RF, P, newDesc, workPassThrough );
  // finish the first scatter
  scatterMatricesS2( halfDesc, reqs3 );
  // start the second scatter
  double *Qsrc2[] = { Q6, Q7 };
  scatterMatricesS0( halfDesc, newDesc, &reqs3, P, work3 );
  scatterMatricesS1( halfDesc, Qsrc2, TPos, 2, newDesc, work3, reqs3 );
  // finish the third gather
  gatherMatricesS2( reqs1, halfDesc, newDesc, work1, LF );
  gatherMatricesS2( reqs2, halfDesc, newDesc, work2, RF );
  // third multiply
  multiplyInternal( LF, RF, P, newDesc, workPassThrough );
  // finish the second scatter
  scatterMatricesS2( halfDesc, reqs3 );
  // start the third scatter
  double *Qsrc3[] = { Q3, Q4 };
  scatterMatricesS0( halfDesc, newDesc, &reqs3, P, work3 );
  scatterMatricesS1( halfDesc, Qsrc3, TPos, 2, newDesc, work3, reqs3 );
  // meanwhile, the last multiply
  multiplyInternal( T5, S5, Q5, halfDesc, workPassThrough );
  // finish the third scatter
  scatterMatricesS2( halfDesc, reqs3 );

  // final calculations of the Q's into C.
  calcCij( numEntriesHalf, C11, C12, C21, C22, Q1, Q2, Q3, Q4, Q5, Q6, Q7 );
  
  deallocate( LF, numEntriesNew );
  deallocate( RF, numEntriesNew );
  deallocate( work1, numEntriesNew );
  deallocate( work2, numEntriesNew );
  deallocate( work3, numEntriesNew );
  deallocate( P, numEntriesNew );
  deallocate( S7, numEntriesHalf );
  deallocate( T5, numEntriesHalf );
  deallocate( S5, numEntriesHalf );
  deallocate( T4, numEntriesHalf );
  deallocate( S4, numEntriesHalf );
  deallocate( T3, numEntriesHalf );
  deallocate( S3, numEntriesHalf );
  deallocate( T6, numEntriesHalf );
  deallocate( Q1, numEntriesHalf );
  deallocate( Q2, numEntriesHalf );
  deallocate( Q3, numEntriesHalf );
  deallocate( Q4, numEntriesHalf );
  deallocate( Q5, numEntriesHalf );
  deallocate( Q6, numEntriesHalf );
  deallocate( Q7, numEntriesHalf );
}

void block_multiply( double *A, double *B, double *C, MatDescriptor d, double *work ) {
  long long lda = d.lda;
  long long lda3 = lda*lda*lda;
  increaseAdditions( lda3 );
  increaseMultiplications( lda3 );
  int nBlocks = d.lda / d.bs;
  // we continue to assume that it divides evenly into blocks
#ifdef SANITY_CHECKS
  assert(nBlocks*d.bs == d.lda);
#endif  
  double *AA, *BB, *CC;
  if( d.bs != 1 && d.bs != d.lda ) {
    // reorder the data into big matrices, but only if it isn't already right
    startTimer(TIMER_REARRANGE);
    AA = work;
    BB = work+d.lda*d.lda;
    CC = work+2*d.lda*d.lda;
    for( int col = 0; col < d.lda; col++ ) {
      int cblock = col / d.bs;
      int cremainder = col-cblock*d.bs;
      for( int rblock = 0; rblock < nBlocks; rblock++ )
	memcpy( AA+col*d.lda+rblock*d.bs, 
		A+(rblock+cblock*nBlocks)*d.bs*d.bs+cremainder*d.bs, 
		d.bs*sizeof(double) );
    }
    for( int col = 0; col < d.lda; col++ ) {
      int cblock = col / d.bs;
      int cremainder = col-cblock*d.bs;
      for( int rblock = 0; rblock < nBlocks; rblock++ )
	memcpy( BB+col*d.lda+rblock*d.bs, 
		B+(rblock+cblock*nBlocks)*d.bs*d.bs+cremainder*d.bs, 
		d.bs*sizeof(double) );
    }
    stopTimer(TIMER_REARRANGE);
  } else {
    AA = A;
    BB = B;
    CC = C;
  }

  // do the multiplication, without requiring CC to be zeroed
  startTimer(TIMER_MUL);
  square_dgemm_zero( d.lda, AA, BB, CC );
  stopTimer(TIMER_MUL);

  // put CC back into C in the correct blocks
  if( d.bs != 1 && d.bs != d.lda ) {
    startTimer(TIMER_REARRANGE);
    for( int col = 0; col < d.lda; col++ ) {
      int cblock = col / d.bs;
      int cremainder = col-cblock*d.bs;
      for( int rblock = 0; rblock < nBlocks; rblock++ )
	memcpy( C+(rblock+cblock*nBlocks)*d.bs*d.bs+cremainder*d.bs,
		CC+col*d.lda+rblock*d.bs,
		d.bs*sizeof(double) );
    }
    stopTimer(TIMER_REARRANGE);
  }
  
}
