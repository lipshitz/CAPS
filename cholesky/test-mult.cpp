#include "mult.h"
#include "communication.h"
#include "library.h"
#include "sizes.h"
#include <stdio.h>
#include <math.h>
#include "counters.h"

// returns the r,c entry of A, where A is stored recursively, full
double getEntrySq( double *A, int n, int r, int c, int nm ) {
  if( n <= nm )
    return A[c*n+r];
  int nhalf = n/2;
  if( r < nhalf && c < nhalf )
    return getEntrySq( A, nhalf, r, c, nm );
  if( c < nhalf )
    return getEntrySq( A+nhalf*nhalf, nhalf, r-nhalf, c, nm );
  if( r < nhalf )
    return getEntrySq( A+2*nhalf*nhalf, nhalf, r, c-nhalf, nm );
  return getEntrySq( A+3*nhalf*nhalf, nhalf, r-nhalf, c-nhalf, nm );
}

// returns the r,c entry of A, where A is stored recursively, packed
double getEntry( double *A, int n, int r, int c, int nm ) {
  if( c > r )
    return getEntry(A, n, c, r, nm);
  if( n <= nm ) {
    if( c <= 0 )
      return A[r];
    return getEntry(A+n,n-1,c-1,r-1,nm);
    //return A[0];
  }
  int nhalf = n/2;
  if( r < nhalf )
    return getEntry(A, nhalf, c, r, nm);
  if( c < nhalf )
    return getEntrySq(A+nhalf*(nhalf+1)/2, nhalf, r-nhalf, c, nm);
  return getEntry(A+nhalf*(nhalf+1)/2+nhalf*nhalf, nhalf, r-nhalf, c-nhalf, nm );
}

void setEntrySq( double *A, int n, int r, int c, double v, int nm ) {
  if( n <= nm ) {
    A[c*n+r] = v;
    //A[0] = v;
    return;
  }
  int nhalf = n/2;
  if( r < nhalf && c < nhalf ) {
    setEntrySq( A, nhalf, r, c, v, nm );
    return;
  }
  if( c < nhalf ) {
    setEntrySq( A+nhalf*nhalf, nhalf, r-nhalf, c, v, nm );
    return;
  }
  if( r < nhalf ) {
    setEntrySq( A+2*nhalf*nhalf, nhalf, r, c-nhalf, v, nm );
    return;
  }
  setEntrySq( A+3*nhalf*nhalf, nhalf, r-nhalf, c-nhalf, v, nm );
  return;
}

// set the r,c entry of A, where A is stored recursively, packed, down to nmin, after which it is packed
void setEntry( double *A, int n, int r, int c, double v, int nm ) {
  if( c > r ) {
    setEntry(A, n, c, r, v, nm);
    return;
  }
  if( n <= nm ) {
    if( c <= 0 ) {
      A[r] = v;
      return;
    }
    setEntry(A+n,n-1,c-1,r-1,v, nm);
    return;
  }
  int nhalf = n/2;
  if( r < nhalf ) {
    setEntry(A, nhalf, c, r, v, nm);
    return;
  }
  if( c < nhalf ) {
    setEntrySq(A+nhalf*(nhalf+1)/2, nhalf, r-nhalf, c, v, nm);
    return;
  }
  setEntry(A+nhalf*(nhalf+1)/2+nhalf*nhalf, nhalf, r-nhalf, c-nhalf, v, nm );
  return;
}


int main( int argc, char **argv ) {

  initCommunication( &argc, &argv );
  initTimers();
  
  // make up a simple test
  int size = read_int( argc, argv, "-s", 8 );
  int r = read_int( argc, argv, "-r", 2 );
  int P;
  MPI_Comm_size( MPI_COMM_WORLD, &P );
  initSizes( P, r, size );
  if( getRank() == 0 ) {
    if( P > (1<<r) )
      printf("Need more recursive steps for this many processors\n");
    if( P > (size/(1<<r))*(size/(1<<r)))
      printf("Need a bigger matrix/fewer recursive steps for this many processors\n");
    printf("-s %d -r %d -n %d\n", size, r, P);
    //printf("%d %d\n", getSizeSq(r,P), getSizeTri(r,P));
  }
  double *A = (double*) malloc( size*size*sizeof(double) );
  double *ACopy = (double*) malloc( size*size*sizeof(double) );
  double *B = (double*) malloc( size*size*sizeof(double) );
  double *BCopy = (double*) malloc( size*size*sizeof(double) );
  double *C = (double*) malloc( size*size*sizeof(double) );
  double *CCopy = (double*) malloc( size*size*sizeof(double) );
  fill(ACopy,size*size);
  fill(BCopy,size*size);
  fill(CCopy,size*size);

  // this assumes that P=4; we set r=2
  // Make X be the recursive version of XCopy, T the recursive version of TCopy
  for( int row = 0; row < size; row++ )
    for( int col = 0; col < size; col++ ) {
      setEntrySq(A, size, row, col, ACopy[row+col*size], size/(1<<r));
      setEntrySq(B, size, row, col, BCopy[row+col*size], size/(1<<r));
      setEntrySq(C, size, row, col, CCopy[row+col*size], size/(1<<r));
    }

  char t = 'T', N = 'N';
  double none = -1., one = 1.;
  dgemm_(&N, &t, &size, &size, &size, &none, ACopy, &size, BCopy, &size, &one, CCopy, &size);
  //if( getRank() == 0 )
  //  printMatrix( CCopy, size );
  //printf("\n");
  double *ADist = (double*) malloc( getSizeSq(r,P)*sizeof(double) );
  double *BDist = (double*) malloc( getSizeSq(r,P)*sizeof(double) );
  double *CDist = (double*) malloc( getSizeSq(r,P)*sizeof(double) );

  distFrom1ProcSq( A, ADist, size, r, P );
  distFrom1ProcSq( B, BDist, size, r, P );
  distFrom1ProcSq( C, CDist, size, r, P );

  mult( CDist, ADist, BDist, size, P, r );

  colTo1ProcSq( C, CDist, size, r, P );
  //printf("\n");
  if( getRank() == 0 ) {
    double maxDiff = 0.;
    if( getRank() == 0 )
      for( int row = 0; row < size; row++ ) {
	for( int col = 0; col < size; col++ ) {
	  //printf( "%7.2g ", getEntrySq(C, size, row, col, size/(1<<r)) );
	  maxDiff = max(maxDiff, fabs((getEntrySq(C,size,row,col,size/(1<<r))-CCopy[row+col*size])/CCopy[row+col*size]));
	}
	//printf("\n");
      }
    printf("Max relative error: %e\n", maxDiff);
  }
  MPI_Finalize();
}
