#include "mult.h"
#include "trsm.h"
#include "communication.h"
#include "sizes.h"
#include "library.h"
#include "counters.h"

void trsm( double *X, double *T, int n, int x, int r ) {
  // We'll support interleaving later
  if( x > 1 && r > 0)
    trsmBFS( X, T, n, x, r );
  else if( x > 1 && r == 0 )
    trsmWasteX( X, T, n, x );
  else if( r == 0 )
    trsmBase( X, T, n );
  else trsmDFS( X, T, n, x, r );
}

// to be called when r = 0, but x > 1
void trsmWasteX( double *X, double *T, int n, int x ) {
  int nOldTri = getSizeTri(0,x);
  int nOldSq = getSizeSq(0,x);
  int rrank = getRelativeRank(x,1);
  double *nX, *nT;
  if( rrank == 0 ) {
    nX = (double*) malloc( x*nOldSq*sizeof(double) );
    nT = (double*) malloc( x*nOldTri*sizeof(double) );
  }
  startTimer(TIMER_COMM_TRSM);
  double *X1[x];
  int Xsizes[x];
  for( int i = 0; i < x; i++ )
    X1[i] = X;
  Xsizes[0] = nOldSq;
  for( int i = 1; i < x; i++ )
    Xsizes[i] = 0;
  reduceBy( x, x, X1, nX, Xsizes );

  double *T1[x];
  int Tsizes[x];
  for( int i = 0; i < x; i++ )
    T1[i] = T;
  Tsizes[0] = nOldTri;
  for( int i = 1; i < x; i++ )
    Tsizes[i] = 0;
  reduceBy( x, x, T1, nT, Tsizes );
  stopTimer(TIMER_COMM_TRSM);

  if( rrank == 0 )
    trsm( nX, nT, n, 1, 0 );

  startTimer(TIMER_COMM_TRSM);
  expandBy( x, x, X1, nX, Xsizes );
  stopTimer(TIMER_COMM_TRSM);
  if( rrank == 0 ) {
    free( nX );
    free( nT );
  }
}

void trsmBase( double *X, double *T, int n ) {
  double *temp = (double*) malloc( n*n*sizeof(double) );
  double *Tp = T;
  startTimer(TIMER_REARRANGE_TRSM);
  for( int c = 0; c < n; c++ )
    for( int r = c; r < n; r++ )
      temp[c*n+r] = *(Tp++);
  stopTimer(TIMER_REARRANGE_TRSM);
  startTimer(TIMER_BASE_TRSM);
  char R = 'R', L = 'L', t = 'T', N = 'N';
  double one = 1.;
  dtrsm_(&R, &L, &t, &N, &n, &n, &one, temp, &n, X, &n);
  stopTimer(TIMER_BASE_TRSM);
  free(temp);
}

void trsmDFS( double *X, double *T, int n, int x, int r ) {
  int nhalf = n/2;
  int nOldTri = getSizeTri(r-1,x);//(nhalf*(nhalf+1)/2+x-1)/x;
  int nOldSq = getSizeSq(r-1,x);//(nhalf*nhalf+x-1)/x;
  double *X11 = X;
  double *X21 = X+nOldSq;
  double *X12 = X21+nOldSq;
  double *X22 = X12+nOldSq;
  double *T11 = T;
  double *T21 = T+nOldTri;
  double *T22 = T21+nOldSq;

  trsm( X11, T11, nhalf, x, r-1 );
  mult( X12, X11, T21, nhalf, x, r-1 );
  trsm( X12, T22, nhalf, x, r-1 );

  trsm( X21, T11, nhalf, x, r-1 );
  mult( X22, X21, T21, nhalf, x, r-1 );
  trsm( X22, T22, nhalf, x, r-1 );
}

void trsmBFS( double *X, double *T, int n, int x, int r ) {
  int nhalf = n/2;

  int xNew = x/2;
  int rrank = getRelativeRank( x, xNew );

  int nOldTri = getSizeTri(r-1,x);//(nhalf*(nhalf+1)/2+x-1)/x;
  int nOldSq = getSizeSq(r-1,x);//(nhalf*nhalf+x-1)/x;

  double *X11 = X;
  double *X21 = X+nOldSq;
  double *X12 = X21+nOldSq;
  double *X22 = X12+nOldSq;
  double *T11 = T;
  double *T21 = T+nOldTri;
  double *T22 = T21+nOldSq;

  double *nX1 = (double*) malloc( nOldSq*2*sizeof(double) );
  double *nX2 = (double*) malloc( nOldSq*2*sizeof(double) );
  double *nT11 = (double*) malloc( nOldTri*2*sizeof(double) );
  double *nT21 = (double*) malloc( nOldSq*2*sizeof(double) );
  double *nT22 = (double*) malloc( nOldTri*2*sizeof(double) );

  // do the re-arrangement
  int sizesSq[] = {nOldSq,nOldSq};
  int sizesTri[] = {nOldTri,nOldTri};
  double *X1[] = {X11,X21};
  double *X2[] = {X12,X22};
  double *T1[] = {T11,T11};
  double *T2[] = {T21,T21};
  double *T3[] = {T22,T22};
  startTimer(TIMER_COMM_TRSM);
  reduceBy( 2, x, X1, nX1, sizesSq );
  reduceBy( 2, x, T1, nT11, sizesTri );
  MPI_Request *req1, *req2;
  double *buf1, *buf2;
  iReduceBy1( 2, x, X2, nX2, sizesSq, req1, buf1 );
  iReduceBy1( 2, x, T2, nT21, sizesSq, req2, buf2 );
  stopTimer(TIMER_COMM_TRSM);

  trsm( nX1, nT11, nhalf, xNew, r-1 );

  startTimer(TIMER_COMM_TRSM);
  iReduceBy2( 2, x, X2, nX2, sizesSq, req1, buf1 );
  iReduceBy2( 2, x, T2, nT21, sizesSq, req2, buf2 );
  iReduceBy1( 2, x, T3, nT22, sizesTri, req1, buf1 );
  stopTimer(TIMER_COMM_TRSM);

  mult( nX2, nX1, nT21, nhalf, xNew, r-1 );

  startTimer(TIMER_COMM_TRSM);
  iReduceBy2( 2, x, T3, nT22, sizesTri, req1, buf1 );
  iExpandBy1( 2, x, X1, nX1, sizesSq, req1, buf1 );
  stopTimer(TIMER_COMM_TRSM);

  trsm( nX2, nT22, nhalf, xNew, r-1 );

  // and the final re-arrangement.  only X needs to be done since T hasn't changed
  startTimer(TIMER_COMM_TRSM);
  iExpandBy2( 2, x, X1, nX1, sizesSq, req1, buf1 );
  expandBy( 2, x, X2, nX2, sizesSq );
  stopTimer(TIMER_COMM_TRSM);
  free(nX1);
  free(nX2);
  free(nT11);
  free(nT21);
  free(nT22);
}

