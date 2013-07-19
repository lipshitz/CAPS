#include "trsm.h"
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
  double *XCopy = (double*) malloc( size*size*sizeof(double) );
  fill(XCopy,size*size);
  double *T = (double*) malloc( size*size*sizeof(double) );
  double *TCopy = (double*) malloc( size*size*sizeof(double) );
  fill(TCopy,size*size);


  // this assumes that P=4; we set r=2
  // Make X be the recursive version of XCopy, T the recursive version of TCopy
  for( int row = 0; row < size; row++ )
    for( int col = 0; col <= row; col++ ) {
      setEntry(T, size, row, col, TCopy[row+col*size], size/(1<<r));
    }
  for( int row = 0; row < size; row++ )
    for( int col = 0; col < size; col++ ) {
      setEntrySq(X, size, row, col, XCopy[row+col*size], size/(1<<r));
    }
  /*
  if( getRank() == 0 ) {
    printMatrix(XCopy, size);
    printf("\n");
    //printMatrix(TCopy, size);
    //printf("\n");
    } */
  char R = 'R', L = 'L', t = 'T', N = 'N';
  double one = 1.;
  dtrsm_(&R, &L, &t, &N, &size, &size, &one, TCopy, &size, XCopy, &size);
  //if( getRank() == 0 )
  //  printMatrix( XCopy, size );
  //printf("\n");
  double *XDist = (double*) malloc( getSizeSq(r,P)*sizeof(double) );
  double *TDist = (double*) malloc( getSizeTri(r,P)*sizeof(double) );
  //if( getRank() == 0 ) {
  //  printf("T:\n");
  //  printMatrix(TCopy,size);
  // }
  /*
  if( getRank() == 0 )
    printf("Sending X\n");
  */
  distFrom1ProcSq( X, XDist, size, r, P );
  //if( getRank() == 0 ) {
  //  printf("XDist: %d\n", getSizeSq(r,P));
  //  printMatrix(XDist,size);
  // }
  /*
  if( getRank() == 0 )
    printf("Sending T\n");
  */
  distFrom1ProcTri( T, TDist, size, r, P );
  //printf("%d TDist: %d\n", getRank(), getSizeTri(r,P));
  //printMatrix(TDist,size);
  /*
  if( getRank() == 0 )
    printf("Performing calculation\n");
  */
  trsm( XDist, TDist, size, P, r );
  //if( getRank() == 0 ) {
  //  printf("XDist: %d\n", getSizeSq(r,P));
  //  printMatrix(XDist,size);
  //}
  /*
  if( getRank() == 0 )
    printf("Receiving X\n");
  */
  colTo1ProcSq( X, XDist, size, r, P );
  //printf("\n");
  if( getRank() == 0 ) {
    double maxDiff = 0.;
    if( getRank() == 0 )
      for( int row = 0; row < size; row++ ) {
	for( int col = 0; col < size; col++ ) {
	  //printf( "%7.2g ", getEntrySq(X, size, row, col, size/(1<<r)) );
	  //printf("%f %f %f\n", getEntrySq(X,size,row,col,size/(1<<r)), XCopy[row+col*size], fabs(getEntrySq(X,size,row,col,size/(1<<r))-XCopy[row+col*size]));
	  maxDiff = max(maxDiff, fabs((getEntrySq(X,size,row,col,size/(1<<r))-XCopy[row+col*size])/XCopy[row+col*size]));
	}
	//printf("\n");
      }
    printf("Max relative error: %e\n", maxDiff);
  }
  MPI_Finalize();
}
