#include "syrk.h"
#include "library.h"
#include "counters.h"

void syrk( double *C, double *A, int n, int x, int r, double alpha ) {
  // We'll support interleaving later
  if( x > 1 && r == 0 )
    syrkWasteX( C, A, n, x, alpha );
  else if( x >= 8 )
    syrkBFS8( C, A, n, x, r, alpha );
  else if( x >= 4 )
    syrkBFS4( C, A, n, x, r, alpha );
  else if( x > 1 )
    syrkBFS2( C, A, n, x, r, alpha );
  else if( r == 0 )
    syrkBase( C, A, n, alpha );
  else
    syrkDFS( C, A, n, x, r, alpha );
}

void syrkBase( double *C, double *A, int n, double alpha ) {
  double *temp = (double*) malloc( n*n*sizeof(double) );
  startTimer(TIMER_REARRANGE_SYRK);
  double *Cp = C;
  for( int c = 0; c < n; c++ )
    for( int r = c; r < n; r++ )
      temp[c*n+r] = *(Cp++);
  char L = 'L', N = 'N';
  double none = -1., one=alpha;
  stopTimer(TIMER_REARRANGE_SYRK);
  startTimer(TIMER_BASE_SYRK);
  dsyrk_(&L, &N, &n, &n, &none, A, &n, &one, temp, &n);
  stopTimer(TIMER_BASE_SYRK);
  startTimer(TIMER_REARRANGE_SYRK);
  Cp = C;
  for( int c = 0; c < n; c++ )
    for( int r = c; r < n; r++ )
      *(Cp++) = temp[c*n+r];
  stopTimer(TIMER_REARRANGE_SYRK);
  free(temp);
}

//  should add alpha=0 optimization to this function
void syrkWasteX( double *C, double *A, int n, int x, double alpha ) {
  int nOldSq = getSizeSq( 0, x );
  int nOldTri = getSizeTri( 0, x );
  int rrank = getRelativeRank(x,1);
  double *nC, *nA;
  if( rrank == 0 ) {
    nC = (double*) malloc( x*nOldTri*sizeof(double) );
    nA = (double*) malloc( x*nOldSq*sizeof(double) );
  }

  startTimer(TIMER_COMM_SYRK);
  int sizesT[x], sizesS[x];
  sizesS[0] = nOldSq;
  sizesT[0] = nOldTri;
  for( int i = 1; i < x; i++ )
    sizesS[i] = 0, sizesT[i] = 0;
  
  double *C1[x];
  for( int i = 0; i < x; i++ )
    C1[i] = C;
  reduceBy( x, x, C1, nC, sizesT );

  double *A1[x];
  for( int i = 0; i < x; i++ )
    A1[i] = A;
  reduceBy( x, x, A1, nA, sizesS );

  stopTimer(TIMER_COMM_SYRK);

  if( rrank == 0 )
    syrk( nC, nA, n, 1, 0, alpha );

  startTimer(TIMER_COMM_SYRK);
  expandBy( x, x, C1, nC, sizesT );
  stopTimer(TIMER_COMM_SYRK);

  if( rrank == 0 ) {
    free( nC );
    free( nA );
  }
}

void syrkDFS( double *C, double *A, int n, int x, int r, double alpha ) {
  int nhalf = n/2;
  int nOldTri = getSizeTri(r-1,x);
  int nOldSq = getSizeSq(r-1,x);
  double *C11 = C;
  double *C21 = C + nOldTri;
  double *C22 = C21 + nOldSq;
  double *A11 = A;
  double *A21 = A+nOldSq;
  double *A12 = A21+nOldSq;
  double *A22 = A12+nOldSq;
  
  syrk( C11, A11, nhalf, x, r-1, alpha );
  syrk( C11, A12, nhalf, x, r-1, 1. );
  mult( C21, A21, A11, nhalf, x, r-1, alpha );
  mult( C21, A22, A12, nhalf, x, r-1, 1. );
  syrk( C22, A21, nhalf, x, r-1, alpha );
  syrk( C22, A22, nhalf, x, r-1, 1. );
}

void syrkBFS4( double *C, double *A, int n, int x, int r, double alpha ) {
  int nhalf = n/2;
  int xNew = x/4;
  int rrank = getRelativeRank(x,xNew);

  int nOldTri = getSizeTri(r-1,x);
  int nOldSq = getSizeSq(r-1,x);
  double *C11 = C;
  double *C21 = C + nOldTri;
  double *C22 = C21 + nOldSq;
  double *A11 = A;
  double *A21 = A+nOldSq;
  double *A12 = A21+nOldSq;
  double *A22 = A12+nOldSq;

  int CSizes[] = {nOldTri,nOldSq,nOldSq,nOldTri};
  int ASizes[] = {nOldSq,nOldSq,nOldSq,nOldSq};
  double *A1[] = {A11,A21,A22,A21};
  double *A2[] = {A12,A11,A12,A22};
  double *lA1 = (double*) malloc( 4*nOldSq*sizeof(double) );
  double *lA2 = (double*) malloc( 4*nOldSq*sizeof(double) );
  double *lC = (double*) malloc( 4*CSizes[rrank]*sizeof(double) );

  startTimer(TIMER_COMM_SYRK);
  reduceBy( 4, x, A1, lA1, ASizes );
  reduceBy( 4, x, A2, lA2, ASizes );
  stopTimer(TIMER_COMM_SYRK);

  if( rrank == 0 || rrank == 3 ) {
    syrk( lC, lA1, nhalf, xNew, r-1, 0. );
    syrk( lC, lA2, nhalf, xNew, r-1, 1. );
  } else {
    mult( lC, lA1, lA2, nhalf, xNew, r-1, 0. );
  }

  double *expC11, *expC21, *expC22;
  if( alpha == 0 ) {
    expC11 = C11;
    expC21 = C21;
    expC22 = C22;
  } else {
    expC11 = (double*) malloc( nOldTri*sizeof(double) );
    expC21 = (double*) malloc( nOldSq*sizeof(double) );
    expC22 = (double*) malloc( nOldTri*sizeof(double) );
  }
  double *cC21 = (double*) malloc( nOldSq*sizeof(double) );
  double *C1[] = {expC11,expC21,cC21,expC22};

  expandBy( 4, x, C1, lC, CSizes );

  int ione = 1;
  double done = 1.;
  if( alpha != 0 ) { // actually, this only works for alpha = 1
    daxpy_( &nOldTri, &done, expC11, &ione, C11, &ione );
    daxpy_( &nOldTri, &done, expC22, &ione, C22, &ione );
    daxpy_( &nOldSq, &done, expC21, &ione, C21, &ione );
    free(expC11);
    free(expC22);
    free(expC21);
  }

  daxpy_( &nOldSq, &done, cC21, &ione, C21, &ione );
  free(lA1);
  free(lA2);
  free(lC);
  free(cC21);
}

void syrkBFS2( double *C, double *A, int n, int x, int r, double alpha ) {
  int nhalf = n/2;
  int xNew = x/2;
  int rrank = getRelativeRank(x,xNew);

  int nOldTri = getSizeTri(r-1,x);
  int nOldSq = getSizeSq(r-1,x);
  double *C11 = C;
  double *C21 = C + nOldTri;
  double *C22 = C21 + nOldSq;
  double *A11 = A;
  double *A21 = A+nOldSq;
  double *A12 = A21+nOldSq;
  double *A22 = A12+nOldSq;

  double *C21c = (double*) malloc( nOldSq*sizeof(double) );
  double *nC1 = (double*) malloc( 2*nOldTri*sizeof(double) );
  double *nC2 = (double*) malloc( 2*nOldSq*sizeof(double) );
  int C1sizes[] = {nOldTri,nOldTri};
  int C2sizes[] = {nOldSq,0};
  int C2sizes2[] = {nOldSq,nOldSq};

  double *A1[] = {A11,A22};
  double *A2[] = {A12,A21};
  double *A3[] = {A21,A12};
  double *nA1 = (double*) malloc( 2*nOldSq*sizeof(double) );
  double *nA2 = (double*) malloc( 2*nOldSq*sizeof(double) );
  double *nA3 = (double*) malloc( 2*nOldSq*sizeof(double) );
  startTimer(TIMER_COMM_SYRK);
  MPI_Request *req;
  double *buf;
  reduceBy( 2, x, A1, nA1, C2sizes2 );
  iReduceBy1( 2, x, A2, nA2, C2sizes2, req, buf );
  stopTimer(TIMER_COMM_SYRK);

  syrk( nC1, nA1, nhalf, xNew, r-1, 0. );

  startTimer(TIMER_COMM_SYRK);
  iReduceBy2( 2, x, A2, nA2, C2sizes2, req, buf );
  iReduceBy1( 2, x, A3, nA3, C2sizes2, req, buf );
  stopTimer(TIMER_COMM_SYRK);

  syrk( nC1, nA2, nhalf, xNew, r-1, 1. );

  double *expC11, *expC21, *expC22;
  if( alpha == 0 ) {
    expC11 = C11;
    expC21 = C21;
    expC22 = C22;
  } else {
    expC11 = (double*) malloc( nOldTri*sizeof(double) );
    expC21 = (double*) malloc( nOldSq*sizeof(double) );
    expC22 = (double*) malloc( nOldTri*sizeof(double) );
  }
  double *C1[] = {expC11,expC22};
  double *C2[] = {expC21,C21c};

  startTimer(TIMER_COMM_SYRK);
  iReduceBy2( 2, x, A3, nA3, C2sizes2, req, buf );
  iExpandBy1( 2, x, C1, nC1, C1sizes, req, buf );
  stopTimer(TIMER_COMM_SYRK);

  if( rrank == 0 )
    mult( nC2, nA3, nA1, nhalf, xNew, r-1, 0. );
  else
    mult( nC2, nA1, nA3, nhalf, xNew, r-1, 0. );


  startTimer(TIMER_COMM_SYRK);
  iExpandBy2( 2, x, C1, nC1, C1sizes, req, buf );
  iExpandBy1( 2, x, C2, nC2, C2sizes2, req, buf );
  stopTimer(TIMER_COMM_SYRK);

  int ione = 1;
  double done = 1.;
  if( alpha != 0 ) { // actually, this only works for alpha = 1
    daxpy_( &nOldTri, &done, expC11, &ione, C11, &ione );
    daxpy_( &nOldTri, &done, expC22, &ione, C22, &ione );
    free(expC11);
    free(expC22);
  }

  startTimer(TIMER_COMM_SYRK);
  iExpandBy2( 2, x, C2, nC2, C2sizes2, req, buf );
  stopTimer(TIMER_COMM_SYRK);

  if( alpha != 0 ) { // actually, this only works for alpha = 1
    daxpy_( &nOldSq, &done, expC21, &ione, C21, &ione );
    free(expC21);
  }

  daxpy_( &nOldSq, &done, C21c, &ione, C21, &ione );

  free(C21c);
  free(nC1);
  free(nC2);
  free(nA1);
  free(nA2);
  free(nA3);
}

void syrkBFS8( double *C, double *A, int n, int x, int r, double alpha ) {
  int nhalf = n/2;
  int xNew = x/4;
  int xNewer = x/8;
  int rrank = getRelativeRank(x,xNew);
  int rrank2 = getRelativeRank(xNew,xNewer);

  int nOldTri = getSizeTri(r-1,x);
  int nOldSq = getSizeSq(r-1,x);
  double *C11 = C;
  double *C21 = C + nOldTri;
  double *C22 = C21 + nOldSq;
  double *A11 = A;
  double *A21 = A+nOldSq;
  double *A12 = A21+nOldSq;
  double *A22 = A12+nOldSq;

  // first do the 4-way re-arrangement.
  int nCSize;
  if( rrank == 0 || rrank == 3 )
    nCSize = 4*nOldTri;
  else
    nCSize = 4*nOldSq;
  double *C21c = (double*) malloc( nOldSq*sizeof(double) );
  //int Csizes[] = {nOldTri,nOldSq,0,nOldTri};
  int Csizes2[] = {nOldTri,nOldSq,nOldSq,nOldTri};
  double *nC = (double*) malloc( 4*Csizes2[rrank]*sizeof(double) );
  //startTimer(TIMER_COMM_SYRK);
  //reduceBy( 4, x, C1, nC, Csizes );
  //stopTimer(TIMER_COMM_SYRK);

  double *A1[] = {A11,A21,A22,A22};
  double *A2[] = {A12,A11,A12,A21};
  int Asizes[] = {nOldSq,nOldSq,nOldSq,nOldSq};
  double *nA1 = (double*) malloc( 4*nOldSq*sizeof(double) );
  double *nA2 = (double*) malloc( 4*nOldSq*sizeof(double) );
  startTimer(TIMER_COMM_SYRK);
  reduceBy( 4, x, A1, nA1, Asizes );
  reduceBy( 4, x, A2, nA2, Asizes );
  stopTimer(TIMER_COMM_SYRK);
  if( rrank == 1 || rrank == 2 ) { // these two do the calls to mult
    mult( nC, nA1, nA2, nhalf, xNew, r-1, 0. );
  } else { // these two will do the recursive syrk calls.  First, we need to split them up further
    double *nCcopy = (double*) malloc( 4*nOldTri*sizeof(double) );
    
    double *nnC = (double*) malloc( 8*nOldTri*sizeof(double) );
    double *nnA = (double*) malloc( 8*nOldSq*sizeof(double) );

    double *nnA1[] = {nA1,nA2};
    int nAsizes[] = {4*nOldSq,4*nOldSq};
    startTimer(TIMER_COMM_SYRK);
    reduceBy( 2, xNew, nnA1, nnA, nAsizes );
    stopTimer(TIMER_COMM_SYRK);
    double *nnC1[] = {nC,nCcopy};
    int nCsizes2[] = {4*nOldTri,4*nOldTri};
    startTimer(TIMER_COMM_SYRK);
    //reduceBy( 2, xNew, nnC1, nnC, nCsizes );
    stopTimer(TIMER_COMM_SYRK);
    syrk( nnC, nnA, nhalf, xNewer, r-1, 0. );
 
    startTimer(TIMER_COMM_SYRK);
    expandBy( 2, xNew, nnC1, nnC, nCsizes2 );
    stopTimer(TIMER_COMM_SYRK);
    // final additions
    int ione = 1;
    double done = 1.;
    int s = 4*nOldTri;
    daxpy_( &s, &done, nCcopy, &ione, nC, &ione );
    free(nCcopy);
    free(nnC);
    free(nnA);
  }
  free(nA1);
  free(nA2);
  // recollect the answers, final additions
  double *expC11, *expC21, *expC22;
  if( alpha == 0. ) {
    expC11 = C11;
    expC21 = C21;
    expC22 = C22;
  } else {
    expC11 = (double*) malloc( nOldTri*sizeof(double) );
    expC21 = (double*) malloc( nOldSq*sizeof(double) );
    expC22 = (double*) malloc( nOldTri*sizeof(double) );
  }
  double *C1[] = {expC11, expC21, C21c, expC22};
  startTimer(TIMER_COMM_SYRK);
  expandBy( 4, x, C1, nC, Csizes2 );
  stopTimer(TIMER_COMM_SYRK);
  free(nC);
  int ione = 1;
  double done = 1.;
  if( alpha != 0 ) { // only correct for alpha=1
    daxpy_( &nOldTri, &done, expC11, &ione, C11, &ione );
    daxpy_( &nOldSq, &done, expC21, &ione, C21, &ione );
    daxpy_( &nOldTri, &done, expC22, &ione, C22, &ione );
    free(expC11);
    free(expC21);
    free(expC22);
  }
  daxpy_( &nOldSq, &done, C21c, &ione, C21, &ione );
  free(C21c);
}
