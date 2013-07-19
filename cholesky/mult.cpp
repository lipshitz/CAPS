#include "mult.h"
#include "communication.h"
#include "sizes.h"
#include "library.h"
#include "counters.h"
#include "pack.cpp"

// this should be tuned.  It is the size below which recursive multiplication is worse than re-ordering then calling BLAS
int minsize = 1024;

void mult( double *C, double *A, double *B, int n, int x, int r, double alpha ) {
  // We'll support interleaving later
  // under some condition:
  if( x > 1 && r == 0 )
    multWasteX( C, A, B, n, x, alpha );
  else if( x >= 8 )
    multBFS8( C, A, B, n, x, r, alpha );
  /*
  else if( x == 4 )
    multBFS4( C, A, B, n, x, r, alpha );
  */
  else if( x > 1 )
    multBFS2( C, A, B, n, x, r, alpha );
  else if( r == 0 || n <= minsize )
    multBase( C, A, B, n, r, alpha );  
  else
    multDFS( C, A, B, n, x, r, alpha );
}

//  should add alpha=0 optimization to this function
void multWasteX( double *C, double *A, double *B, int n, int x, double alpha ) {
  int nOldSq = getSizeSq( 0, x );
  int rrank = getRelativeRank( x, 1 );
  double *nA, *nB, *nC;
  if( rrank == 0 ) {
    nC = (double*) malloc( x*nOldSq*sizeof(double) );
    nA = (double*) malloc( x*nOldSq*sizeof(double) );
    nB = (double*) malloc( x*nOldSq*sizeof(double) );
  }

  startTimer(TIMER_COMM_MULT);
  int sizes[x];
  sizes[0] = nOldSq;
  for( int i = 1; i < x; i++ )
    sizes[i] = 0;

  double *C1[x];
  for( int i = 0; i < x; i++ )
    C1[i] = C;
  reduceBy( x, x, C1, nC, sizes );

  double *A1[x];
  for( int i = 0; i < x; i++ )
    A1[i] = A;
  reduceBy( x, x, A1, nA, sizes );

  double *B1[x];
  for( int i = 0; i < x; i++ )
    B1[i] = B;
  reduceBy( x, x, B1, nB, sizes );
  stopTimer(TIMER_COMM_MULT);
  if( rrank == 0 )
    mult( nC, nA, nB, n, 1, 0, alpha );

  startTimer(TIMER_COMM_MULT);
  expandBy( x, x, C1, nC, sizes );
  stopTimer(TIMER_COMM_MULT);

  if( rrank == 0 ) {
    free(nC);
    free(nA);
    free(nB);
  }
}

void multBase( double *C, double *A, double *B, int n, int r, double alpha ) {
  double *AA,*BB,*CC;
  int bs;
  if( r == 0 ) {
    AA = A;
    BB = B;
    CC = C;
  } else {
    startTimer(TIMER_REARRANGE_MULT);
    AA = (double*) malloc( n*n*sizeof(double) );
    BB = (double*) malloc( n*n*sizeof(double) );
    CC = (double*) malloc( n*n*sizeof(double) );    
    bs = n/(1<<r);
    unpack( n, AA, A, bs );
    unpack( n, BB, B, bs );
    if( alpha != 0 )
      unpack( n, CC, C, bs );
    stopTimer(TIMER_REARRANGE_MULT);
  }
  char N = 'N', T = 'T';
  double none = -1.;
  startTimer(TIMER_BASE_MULT);
  dgemm_(&N, &T, &n, &n, &n, &none, AA, &n, BB, &n, &alpha, CC, &n);
  stopTimer(TIMER_BASE_MULT);
  if( r != 0 ) {
    startTimer(TIMER_REARRANGE_MULT);
    pack( n, CC, C, bs );
    free(AA);
    free(BB);
    free(CC);
    stopTimer(TIMER_REARRANGE_MULT);
  }
}

void multDFS( double *C, double *A, double *B, int n, int x, int r, double alpha ) {
  int nhalf = n/2;
  int nOldSq = getSizeSq( r-1, x );//(nhalf*nhalf+x-1)/x;

  double *C11 = C;
  double *C21 = C+nOldSq;
  double *C12 = C21+nOldSq;
  double *C22 = C12+nOldSq;

  double *A11 = A;
  double *A21 = A+nOldSq;
  double *A12 = A21+nOldSq;
  double *A22 = A12+nOldSq;

  double *B11 = B;
  double *B21 = B+nOldSq;
  double *B12 = B21+nOldSq;
  double *B22 = B12+nOldSq;

  mult( C11, A11, B11, nhalf, x, r-1, alpha );
  mult( C11, A12, B12, nhalf, x, r-1, 1. );
  mult( C12, A11, B21, nhalf, x, r-1, alpha );
  mult( C12, A12, B22, nhalf, x, r-1, 1. );
  mult( C21, A21, B11, nhalf, x, r-1, alpha );
  mult( C21, A22, B12, nhalf, x, r-1, 1. );
  mult( C22, A21, B21, nhalf, x, r-1, alpha );
  mult( C22, A22, B22, nhalf, x, r-1, 1. );

}

void multBFS8( double *C, double *A, double *B, int n, int x, int r, double alpha ) {
  int nhalf = n/2;
  int nOldSq = getSizeSq( r-1, x );

  double *C11 = C;
  double *C21 = C+nOldSq;
  double *C12 = C21+nOldSq;
  double *C22 = C12+nOldSq;

  double *A11 = A;
  double *A21 = A+nOldSq;
  double *A12 = A21+nOldSq;
  double *A22 = A12+nOldSq;

  double *B11 = B;
  double *B21 = B+nOldSq;
  double *B12 = B21+nOldSq;
  double *B22 = B12+nOldSq;

  int xNew = x/8;
  int rrank = getRelativeRank( x, xNew );

  double *C11c = (double*) malloc( nOldSq*sizeof(double) );
  double *C12c = (double*) malloc( nOldSq*sizeof(double) );
  double *C21c = (double*) malloc( nOldSq*sizeof(double) );
  double *C22c = (double*) malloc( nOldSq*sizeof(double) );

  double *LA = (double*) malloc( 8*nOldSq*sizeof(double) );
  double *LB = (double*) malloc( 8*nOldSq*sizeof(double) );
  double *LC = (double*) malloc( 8*nOldSq*sizeof(double) );

  int sizes1[] = {nOldSq,  0,  nOldSq,  0,  nOldSq,  0,  nOldSq,  0};
  int sizes2[] = {nOldSq,  nOldSq,  nOldSq,  nOldSq,  nOldSq,  nOldSq,  nOldSq,  nOldSq};
  startTimer(TIMER_COMM_MULT);
  //reduceBy( 8, x, Cargs, LC, sizes1 );
  double *Aargs[] = {A11, A12, A11, A12, A21, A22, A21, A22};
  reduceBy( 8, x, Aargs, LA, sizes2 );
  double *Bargs[] = {B11, B12, B21, B22, B11, B12, B21, B22};
  reduceBy( 8, x, Bargs, LB, sizes2 );
  stopTimer(TIMER_COMM_MULT);

  // perform the recursive calculations

  mult( LC, LA, LB, nhalf, xNew, r-1, 0. );

  // re-arrange C and Cc
  double *eC11, *eC12, *eC21, *eC22;
  if( alpha == 0. ) {
    eC11 = C11;
    eC12 = C12;
    eC21 = C21;
    eC22 = C22;
  } else {
    eC11 = (double*) malloc( nOldSq*sizeof(double) );
    eC12 = (double*) malloc( nOldSq*sizeof(double) );
    eC21 = (double*) malloc( nOldSq*sizeof(double) );
    eC22 = (double*) malloc( nOldSq*sizeof(double) );
  }
  double *Cargs[] = {eC11, C11c, eC12, C12c, eC21, C21c, eC22, C22c};
  startTimer(TIMER_COMM_MULT);
  expandBy( 8, x, Cargs, LC, sizes2 );
  stopTimer(TIMER_COMM_MULT);
  
  // and the final additions
  int ione = 1;
  double done = 1.;
  if( alpha != 0. ) {// actually only works if 1.
    daxpy_( &nOldSq, &done, eC11 , &ione, C11, &ione );
    daxpy_( &nOldSq, &done, eC12 , &ione, C12, &ione );
    daxpy_( &nOldSq, &done, eC21 , &ione, C21, &ione );
    daxpy_( &nOldSq, &done, eC22 , &ione, C22, &ione );
    free(eC11);
    free(eC12);
    free(eC21);
    free(eC22);
  }
  daxpy_( &nOldSq, &done, C11c, &ione, C11, &ione );
  daxpy_( &nOldSq, &done, C12c, &ione, C12, &ione );
  daxpy_( &nOldSq, &done, C21c, &ione, C21, &ione );
  daxpy_( &nOldSq, &done, C22c, &ione, C22, &ione );

  free(C11c);
  free(C12c);
  free(C21c);
  free(C22c);
  free(LA);
  free(LB);
  free(LC);
}
/*
void multBFS4( double *C, double *A, double *B, int n, int x, int r ) {
  int nhalf = n/2;
  int nOldSq = getSizeSq( r-1, x );//(nhalf*nhalf+x-1)/x;

  double *C11 = C;
  double *C21 = C+nOldSq;
  double *C12 = C21+nOldSq;
  double *C22 = C12+nOldSq;

  double *A11 = A;
  double *A21 = A+nOldSq;
  double *A12 = A21+nOldSq;
  double *A22 = A12+nOldSq;

  double *B11 = B;
  double *B21 = B+nOldSq;
  double *B12 = B21+nOldSq;
  double *B22 = B12+nOldSq;

  int xNew = x/4;
  int rrank = getRelativeRank( x, xNew );

  double *LA = (double*) malloc( 4*nOldSq*sizeof(double) );
  double *LB = (double*) malloc( 4*nOldSq*sizeof(double) );
  double *LC = (double*) malloc( 4*nOldSq*sizeof(double) );

  reduceBy4( x, C11, C12, C21, C22, LC, nOldSq );
  reduceBy4( x, A11, A11, A21, A21, LA, nOldSq );
  reduceBy4( x, B11, B21, B11, B21, LB, nOldSq );

  mult( LC, LA, LB, nhalf, xNew, r-1 );

  reduceBy4( x, A12, A12, A22, A22, LA, nOldSq );
  reduceBy4( x, B12, B22, B12, B22, LB, nOldSq );

  mult( LC, LA, LB, nhalf, xNew, r-1 );

  expandBy4( C11, C12, C21, C22, LC, nOldSq );  

  free( LA );
  free( LB );
  free( LC );
}
*/
void multBFS2( double *C, double *A, double *B, int n, int x, int r, double alpha ) {
  int nhalf = n/2;
  int nOldSq = getSizeSq(r-1,x);

  double *C11 = C;
  double *C21 = C+nOldSq;
  double *C12 = C21+nOldSq;
  double *C22 = C12+nOldSq;

  double *A11 = A;
  double *A21 = A+nOldSq;
  double *A12 = A21+nOldSq;
  double *A22 = A12+nOldSq;

  double *B11 = B;
  double *B21 = B+nOldSq;
  double *B12 = B21+nOldSq;
  double *B22 = B12+nOldSq;

  int xNew = x/2;
  int rrank = getRelativeRank( x, xNew );

  double *LA = (double*) malloc( 2*nOldSq*sizeof(double) );
  double *LA2 = (double*) malloc( 2*nOldSq*sizeof(double) );
  double *LB = (double*) malloc( 2*nOldSq*sizeof(double) );
  double *LB2 = (double*) malloc( 2*nOldSq*sizeof(double) );
  double *LC = (double*) malloc( 2*nOldSq*sizeof(double) );
  double *LC2 = (double*) malloc( 2*nOldSq*sizeof(double) );

  int sizes[] = {nOldSq,  nOldSq};
  startTimer(TIMER_COMM_MULT);
  // perform the communication for the first multiplication
  double *A1[] = {A11,A11};
  reduceBy( 2, x, A1, LA, sizes );
  double *B1[] = {B11,B21};
  reduceBy( 2, x, B1, LB, sizes );
  // start the communication for the second
  double *A2[] = {A12,A12};
  double *B2[] = {B12,B22};
  MPI_Request *reqA;
  double *bufA;
  MPI_Request *reqB;
  double *bufB;
  iReduceBy1( 2, x, A2, LA2, sizes, reqA, bufA );
  iReduceBy1( 2, x, B2, LB2, sizes, reqB, bufB );
  stopTimer(TIMER_COMM_MULT);

  // perform the first multiplication
  mult( LC, LA, LB, nhalf, xNew, r-1, 0. );

  startTimer(TIMER_COMM_MULT);
  // finish the commuication for the second
  iReduceBy2( 2, x, A2, LA2, sizes, reqA, bufA );
  iReduceBy2( 2, x, B2, LB2, sizes, reqB, bufB );
  stopTimer(TIMER_COMM_MULT);
  // start the communication for the third
  double *A3[] = {A21, A21};
  iReduceBy1( 2, x, A3, LA, sizes, reqA, bufA );
  // perform the second multiplication
  mult( LC, LA2, LB2, nhalf, xNew, r-1, 1. );

  double *eC11, *eC12;
  if( alpha == 0. ) {
    eC11 = C11;
    eC12 = C12;
  } else {
    eC11 =  (double*) malloc( nOldSq*sizeof(double) );
    eC12 =  (double*) malloc( nOldSq*sizeof(double) );
  }

  startTimer(TIMER_COMM_MULT);
  // finish the communication for the third
  iReduceBy2( 2, x, A3, LA, sizes, reqA, bufA );
  // start the communication for the fourth
  double *A4[] = {A22,A22};
  iReduceBy1( 2, x, A4, LA2, sizes, reqA, bufA );
  stopTimer(TIMER_COMM_MULT);

  // perform the third multiplication
  mult( LC2, LA, LB, nhalf, xNew, r-1, 0. );
  startTimer(TIMER_COMM_MULT);
  // finish the communication for the fourth
  iReduceBy2( 2, x, A4, LA2, sizes, reqA, bufA );
  // start gathering the results of the first two
  double *C1[] = {eC11,eC12};
  iExpandBy1( 2, x, C1, LC, sizes, reqB, bufB );
  stopTimer(TIMER_COMM_MULT);

  // perform the fourth multiplication
  mult( LC2, LA2, LB2, nhalf, xNew, r-1, 1. );

  // finish additions from the first two
  iExpandBy2( 2, x, C1, LC, sizes, reqB, bufB );
  int ione = 1;
  double done = 1.;
  if( alpha == 0. ) {
    eC11 = C21;
    eC12 = C22;
  } else {
    daxpy_( &nOldSq, &done, eC11 , &ione, C11, &ione );
    daxpy_( &nOldSq, &done, eC12 , &ione, C12, &ione );
  }

  // gather and add the last two
  startTimer(TIMER_COMM_MULT);
  double *C2[] = {eC11,eC12};
  expandBy( 2, x, C2, LC2, sizes );
  stopTimer(TIMER_COMM_MULT);

  if( alpha != 0. ) {
    daxpy_( &nOldSq, &done, eC11 , &ione, C21, &ione );
    daxpy_( &nOldSq, &done, eC12 , &ione, C22, &ione );
    free( eC11 );
    free( eC12 );
  }
  free(LA);
  free(LA2);
  free(LB);
  free(LB2);
  free(LC);
  free(LC2);
}
