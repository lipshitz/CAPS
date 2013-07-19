#include "rectmm.h"
#include "communication.h"
#include "rectsizes.h"
#include "blas.h"

/*
  Compute C = A * B
  m = #rows of A,C
  n = #columns of B,C
  k = #columns of A, #rows of B
  P = #processors involved
  r = #recursive steps
  patt = array of 'B' and 'D' indicating BFS or DFS steps; length r
  divPatt = array of how much each of m,n,k are divided by at each recursive step; length 3*r

  The layout is: recursive with number of blocks at each level given by divPatt down to some base-case size, in column-major order; then element-wise cyclic in column-major order.  If the base-case size isn't a multiple of P (initial), then there will may be padding, but it is always at the end of the array.
 */

// this is just a wrapper to initialize and call the recursive function
void rectMM( double *A, double *B, double *C, int m, int n, int k, int P, int r, char *patt, int *divPatt ) {
  //initSizesRect( m, n, k, P, r, divPatt );
  recRectMM( A, B, C, m, n, k, P, r, patt, divPatt );
}

// dispatch to local call, BFS, or DFS as appropriate
void recRectMM( double *A, double *B, double *C, int m, int n, int k, int P, int r, char *patt, int *divPatt ) {
  if( r == 0 ) {
    if( P > 1 ) {
      printf("Error: reached r=0 with more than one processor\n");
      exit(-1);
    }
    localRectMM( A, B, C, m, n, k );
  } else {
    if( patt[0] == 'B' || patt[0] == 'b' )
      rectBFS( A, B, C, m, n, k, P, r, divPatt[0], divPatt[1], divPatt[2], patt+1, divPatt+3 );
    else if( patt[0] == 'D' || patt[0] == 'd' )
      rectDFS( A, B, C, m, n, k, P, r, divPatt[0], divPatt[1], divPatt[2], patt+1, divPatt+3 );
    else {
      printf("Error: unrecognized type of step: %c\n", patt[0] );
    }
  }
}

void localRectMM( double *A, double *B, double *C, int m, int n, int k ) {
  char N = 'N';
  double one = 1., zero = 0.;
  dgemm_( &N, &N, &m, &n, &k, &one, A, &m, B, &k, &zero, C, &m );
}

void rectDFS( double *A, double *B, double *C, int m, int n, int k, int P, int r, int divm, int divn, int divk, char *patt, int *divPatt ) {
  int newm = m/divm;
  int newn = n/divn;
  int newk = k/divk;
  int subAsize = getSizeRect( newm, 1, newk, P );
  int subBsize = getSizeRect( 1, newn, newk, P );
  int subCsize = getSizeRect( newm, newn, 1, P );
  double *As = A, *Bs = B, *Cs = C;
  double *Ap, *Cp;
  for( int N = 0; N < divn; N++ ) {
    Ap = A;
    for( int K = 0; K < divk; K++ ) {
      Cp = C;
      for( int M = 0; M < divm; M++ ) {
	recRectMM( Ap, B, Cp, newm, newn, newk, P, r-1, patt, divPatt );
	Ap += subAsize;
	Cp += subCsize;
      }
      B += subBsize;
    }
    C += subCsize;
  }
}

void rectBFS( double *A, double *B, double *C, int m, int n, int k, int P, int r, int divm, int divn, int divk, char *patt, int *divPatt ) {
  int newm = m/divm;
  int newn = n/divn;
  int newk = k/divk;
  int sub = divm*divn*divk;
  int newP = P/sub;
  int subAsize = getSizeRect( newm, 1, newk, P );
  int subBsize = getSizeRect( 1, newn, newk, P );
  int subCsize = getSizeRect( newm, newn, 1, P );
  int newAsize = getSizeRect( newm, 1, newk, newP );
  int newBsize = getSizeRect( 1, newn, newk, newP );
  int newCsize = getSizeRect( newm, newn, 1, newP );
  int Csize = getSizeRect( m, n, 1, P );

  double *Ccopies = (double*) malloc( sizeof(double)*Csize*(divk-1) );

  int Asizes[sub];
  int Bsizes[sub];
  int Csizes[sub];
  for( int i = 0; i < sub; i++ ) {
    Asizes[i] = subAsize;
    Bsizes[i] = subBsize;
    Csizes[i] = subCsize;
  }
  double *Aargs[sub];
  double *Bargs[sub];
  double *Cargs[sub];
  int count = 0;
  for( int N = 0; N < divn; N++ )
    for( int K = 0; K < divk; K++ )
      for( int M = 0; M < divm; M++, count++ ) {
	Aargs[count] = A+(M+K*divm)*subAsize;
	Bargs[count] = B+(K+N*divk)*subBsize;
	if( K == 0 )
	  Cargs[count] = C+(M+N*divm)*subCsize;
	else
	  Cargs[count] = Ccopies+(M+N*divm)*subCsize+(K-1)*Csize;
      }

  double *LA = (double*) malloc( sizeof(double)*newAsize );
  double *LB = (double*) malloc( sizeof(double)*newBsize );
  double *LC = (double*) malloc( sizeof(double)*newCsize );

  reduceBy( sub, P, Aargs, LA, Asizes );
  reduceBy( sub, P, Bargs, LB, Bsizes );

  recRectMM( LA, LB, LC, newm, newn, newk, newP, r-1, patt, divPatt );

  expandBy( sub, P, Cargs, LC, Csizes );

  // final additions; could use some optimization
  int i_one = 1;
  double one = 1.;
  for( int K = 1; K < divk; K++ )
    daxpy_( &Csize, &one, Ccopies+(K-1)*Csize, &i_one, C, &i_one );

  free(Ccopies);
  free(LA);
  free(LB);
  free(LC);

}
