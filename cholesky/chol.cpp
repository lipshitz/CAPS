#include "chol.h"
#include "counters.h"

void chol( double *A, int n, int x, int r ) {
  if( x == 1 && r == 0 )
    cholBase( A, n );
  else if( x < (1<<(3*r)) )
    cholDFS( A, n, x ,r );
  else if( x == (1<<(3*r)) )
    cholWaste2S( A, n, x, r );
  else
    cholWaste2( A, n, x, r );
}

void cholBase( double *A, int n ) {
  int info = 0;
  // this uses the unpacked, but blocked version.
  double *temp = (double*) malloc( n*n*sizeof(double) );
  double *Ap = A;
  startTimer(TIMER_REARRANGE_CHOL);
  for( int c = 0; c < n; c++ )
    for( int r = c; r < n; r++ )
      temp[c*n+r] = *(Ap++);
  stopTimer(TIMER_REARRANGE_CHOL);
  startTimer(TIMER_BASE_CHOL);
  char L = 'L', N = 'N';
  double none = -1., one = 1.;
  dpotrf_( &L, &n, temp, &n, &info);
  if( info != 0 )
    printf("Info is %d, don't trust any results.\n", info);
  stopTimer(TIMER_BASE_CHOL);
  startTimer(TIMER_REARRANGE_CHOL);
  Ap = A;
  for( int c = 0; c < n; c++ )
    for( int r = c; r < n; r++ )
      *(Ap++) = temp[c*n+r];
  stopTimer(TIMER_REARRANGE_CHOL);
  free(temp);
}

void cholDFS( double *A, int n, int x, int r ) {
  int nhalf = n/2;
  int nOldTri = getSizeTri(r-1,x);
  int nOldSq = getSizeSq(r-1,x);
  double *A11 = A;
  double *A21 = A+nOldTri;
  double *A22 = A21+nOldSq;
  chol(A11, nhalf, x, r-1);
  trsm(A21, A11, nhalf, x, r-1);
  syrk(A22, A21, nhalf, x, r-1);
  chol(A22, nhalf, x, r-1);
}

void cholWaste2( double *A, int n, int x, int r ) {
  int full = getSizeTri(r,x);
  int xNew = x/2;
  int rrank = getRelativeRank(x,xNew);

  double *A1[] = {A,A};
  int sizes[] = {full,0};
  double *nA = (double*) malloc( full*2*sizeof(double) );

  startTimer(TIMER_COMM_CHOL);
  reduceBy( 2, x, A1, nA, sizes );
  stopTimer(TIMER_COMM_CHOL);
  if( rrank == 0 )
    chol( nA, n, xNew, r );
  startTimer(TIMER_COMM_CHOL);
  expandBy( 2, x, A1, nA, sizes );
  stopTimer(TIMER_COMM_CHOL);
  free( nA );
}

// like the above, but actually takes a step.  as a result, overlap is possible
void cholWaste2S( double *A, int n, int x, int r ) {
  int nhalf = n/2;
  int nOldTri = getSizeTri(r-1,x);
  int nOldSq = getSizeSq(r-1,x);
  double *A11 = A;
  double *A21 = A+nOldTri;
  double *A22 = A21+nOldSq;
  int xNew = x/2;
  int rrank = getRelativeRank(x,xNew);

  double *A1[] = {A11,A11};
  double *A2[] = {A21,A21};
  double *A3[] = {A22,A22};

  int sizes1[] = {nOldTri,0};
  int sizes2[] = {nOldSq,0};

  double *nA11, *nA21, *nA22;
  if( rrank == 0 ) {
    nA11 = (double*) malloc( nOldTri*2*sizeof(double) );
    nA21 = (double*) malloc( nOldSq*2*sizeof(double) );
    nA22 = (double*) malloc( nOldTri*2*sizeof(double) );
  }

  MPI_Request *req1, *req2;
  double *buf1, *buf2;

  startTimer(TIMER_COMM_CHOL);
  reduceBy( 2, x, A1, nA11, sizes1 );
  iReduceBy1( 2, x, A2, nA21, sizes2, req1, buf1 );
  stopTimer(TIMER_COMM_CHOL);

  if( rrank == 0 )
    chol(nA11, nhalf, xNew, r-1);

  startTimer(TIMER_COMM_CHOL);
  iReduceBy2( 2, x, A2, nA21, sizes2, req1, buf1 );
  iReduceBy1( 2, x, A3, nA22, sizes1, req1, buf1 );
  iExpandBy1( 2, x, A1, nA11, sizes1, req2, buf2 );
  stopTimer(TIMER_COMM_CHOL);

  if( rrank == 0 )
    trsm(nA21, nA11, nhalf, xNew, r-1);    

  startTimer(TIMER_COMM_CHOL);
  iReduceBy2( 2, x, A3, nA22, sizes1, req1, buf1 );
  iExpandBy1( 2, x, A2, nA21, sizes2, req1, buf1 );
  stopTimer(TIMER_COMM_CHOL);

  if( rrank == 0 ) {
    syrk(nA22, nA21, nhalf, xNew, r-1);
    chol(nA22, nhalf, xNew, r-1);
  }

  startTimer(TIMER_COMM_CHOL);
  iExpandBy2( 2, x, A1, nA11, sizes1, req2, buf2 );
  expandBy( 2, x, A3, nA22, sizes1 );
  iExpandBy2( 2, x, A2, nA21, sizes2, req1, buf1 );
  stopTimer(TIMER_COMM_CHOL);

  if( rrank == 0 ) {
    free(nA11);
    free(nA21);
    free(nA22);
  }
}
