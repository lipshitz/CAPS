#include "chol.h"
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
    if( P > (size/(1<<r))*(size/(1<<r)+1)/2)
      printf("Need a bigger matrix/fewer recursive steps for this many processors\n");
    printf("-s %d -r %d -n %d\n", size, r, P);
  }
  double *X = (double*) malloc( size*size*sizeof(double) );
  fill(X,size*size);
  double *A = (double*) malloc( size*size*sizeof(double) );
  double *ACopy = (double*) malloc( size*size*sizeof(double) );
  char T = 'T', N = 'N';
  double one = 1., zero = 0.;
  dgemm_( &T, &N, &size, &size, &size, &one, X, &size, X, &size, &zero, ACopy, &size );

  for( int row = 0; row < size; row++ )
    for( int col = 0; col <= row; col++ ) {
      setEntry(A, size, row, col, ACopy[row+col*size], size/(1<<r));
    }
  char L = 'L';
  int info;
  dpotrf_( &L, &size, ACopy, &size, &info);
  double *ADist = (double*) malloc( getSizeTri(r,P)*sizeof(double) );
  distFrom1ProcTri( A, ADist, size, r, P );
  chol( ADist, size, P, r );
  colTo1ProcTri( A, ADist, size, r, P );
  if( getRank() == 0 ) {
    double maxDiff = 0.;
    if( getRank() == 0 )
      for( int row = 0; row < size; row++ ) {
	for( int col = 0; col <= row; col++ ) {
	  maxDiff = max(maxDiff, fabs((getEntry(A,size,row,col,size/(1<<r))-ACopy[row+col*size])/ACopy[row+col*size]));
	}
      }
    printf("Max relative error: %e\n", maxDiff);
  }
  MPI_Finalize();
}
