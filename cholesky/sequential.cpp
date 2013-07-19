#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define max(a,b) (((a)>(b))?(a):(b))

extern "C" {
  void dpotrf_( char*, int*, double*, int*, int* );
  void dpptrf_( char*, int*, double*, int* );
  void dgemm_( char*, char*, int*,int*,int*, double*, double*,int*, double*,int*, double*, double*,int*);
  void dtrsm_(char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int*);
  void dsyrk_( char*, char*, int*, int*, double*, double*, int*, double*, double*, int* );
}

int nmin = 64;

double read_timer( )
{
  static bool initialized = false;
  static struct timeval start;
  struct timeval end;
  if( !initialized )
    {
      gettimeofday( &start, NULL );
      initialized = true;
    }

  gettimeofday( &end, NULL );

  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

void fill( double *p, int n ) {
  for( int i = 0; i < n; i++ )
    p[i] = 2*drand48()-1;
}

// C -= A*B^t
void mult( double *C, double *A, double *B, int n ) {
  if( n <= nmin ) {
    char N = 'N', T = 'T';
    double none = -1., one = 1.;
    dgemm_(&N, &T, &n, &n, &n, &none, A, &n, B, &n, &one, C, &n);
    return;
  }
  int nhalf = n/2;

  double *C11 = C;
  double *C21 = C+nhalf*nhalf;
  double *C12 = C21+nhalf*nhalf;
  double *C22 = C12+nhalf*nhalf;

  double *A11 = A;
  double *A21 = A+nhalf*nhalf;
  double *A12 = A21+nhalf*nhalf;
  double *A22 = A12+nhalf*nhalf;

  double *B11 = B;
  double *B21 = B+nhalf*nhalf;
  double *B12 = B21+nhalf*nhalf;
  double *B22 = B12+nhalf*nhalf;

  mult( C11, A11, B11, nhalf );
  mult( C11, A12, B12, nhalf );
  mult( C12, A11, B21, nhalf );
  mult( C12, A12, B22, nhalf );
  mult( C21, A21, B11, nhalf );
  mult( C21, A22, B12, nhalf );
  mult( C22, A21, B21, nhalf );
  mult( C22, A22, B22, nhalf );
  
}

void printMatrix(double*, int);

// solve X_out*T^t=X_in
void trsm( double *X, double *T, int n ) {
  // base case
  if( n <= nmin ) {
    // copy into and out of a temporary full matrix; is there really no packed trsm?
    double *temp = (double*) malloc( n*n*sizeof(double) );
    double *Tp = T;
    for( int c = 0; c < n; c++ )
      for( int r = c; r < n; r++ )
	temp[c*n+r] = *(Tp++);
    char R = 'R', L = 'L', T = 'T', N = 'N';
    double one = 1.;
    dtrsm_(&R, &L, &T, &N, &n, &n, &one, temp, &n, X, &n);
    free(temp);
    return;
  }
  int nhalf = n/2;
  double *X11 = X;
  double *X21 = X+nhalf*nhalf;
  double *X12 = X21+nhalf*nhalf;
  double *X22 = X12+nhalf*nhalf;
  double *T11 = T;
  double *T21 = T+nhalf*(nhalf+1)/2;
  double *T22 = T21+nhalf*nhalf;

  trsm( X11, T11, nhalf );
  mult( X12, X11, T21, nhalf ); // This will do X12 = X12-X11*T21^t
  trsm( X12, T22, nhalf );

  // the above and below triplets should be able to be done completely independently of eachother

  trsm( X21, T11, nhalf );
  mult( X22, X21, T21, nhalf ); // This will do X22 = X22-X21*T21^t
  trsm( X22, T22, nhalf );
}

// computes C -= A*A^t, where C is symmetric, half stored, A is general
void syrk( double *C, double *A, int n ) {
  // base case
  if( n <= nmin ) {
    double *temp = (double*) malloc( n*n*sizeof(double) );
    double *Cp = C;
    for( int c = 0; c < n; c++ )
      for( int r = c; r < n; r++ )
	temp[c*n+r] = *(Cp++);
    char L = 'L', N = 'N';
    double none = -1., one = 1.;
    dsyrk_(&L, &N, &n, &n, &none, A, &n, &one, temp, &n);
    Cp = C;
    for( int c = 0; c < n; c++ )
      for( int r = c; r < n; r++ )
	*(Cp++) = temp[c*n+r];
    free(temp);
    //C[0] -= A[0]*A[0];
    return;
  }
  int nhalf = n/2;
  double *C11 = C;
  double *C21 = C + nhalf*(nhalf+1)/2;
  double *C22 = C21 + nhalf*nhalf;
  double *A11 = A;
  double *A21 = A+nhalf*nhalf;
  double *A12 = A21+nhalf*nhalf;
  double *A22 = A12+nhalf*nhalf;
  
  // these can be made independent with the use of some intermediates, and some final additions
  syrk( C11, A11, nhalf );
  syrk( C11, A12, nhalf );
  mult( C21, A21, A11, nhalf ); // This will do C21 = C21-A21*A11^t
  mult( C21, A22, A12, nhalf );
  syrk( C22, A21, nhalf );
  syrk( C22, A22, nhalf );
}

// assume that A, which is triangular, is stored in recursive L; that means that the square block is stored in recursive backwards N
void chol( double *A, int n ) {
  // base case
  if( n <= nmin ) {
    // probably we want to copy into full, since there doesn't seem to be a blocked packed cholesky in lapack; but the easy version for now
    int info = 0;
    //char L = 'L';
    //dpotrf_( &L, &size, Afull, &size, &info);
    //dpptrf_( &L, &n, A, &info);
    //A[0] = sqrt(A[0]);
    // this uses the unpacked, but blocked version.
    double *temp = (double*) malloc( n*n*sizeof(double) );
    double *Ap = A;
    for( int c = 0; c < n; c++ )
      for( int r = c; r < n; r++ )
	temp[c*n+r] = *(Ap++);
    char L = 'L', N = 'N';
    double none = -1., one = 1.;
    dpotrf_( &L, &n, temp, &n, &info);
    Ap = A;
    for( int c = 0; c < n; c++ )
      for( int r = c; r < n; r++ )
	*(Ap++) = temp[c*n+r];
    free(temp);
    return;
  }
  int nhalf = n/2;
  double *A11 = A;
  double *A21 = A+nhalf*(nhalf+1)/2;
  double *A22 = A21+nhalf*nhalf;
  chol(A11,nhalf);
  trsm(A21,A11,nhalf);
  syrk(A22,A21,nhalf);
  chol(A22,nhalf);
}

// returns the r,c entry of A, where A is stored recursively, full
double getEntrySq( double *A, int n, int r, int c, int nm=nmin ) {
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
double getEntry( double *A, int n, int r, int c, int nm=nmin ) {
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

void setEntrySq( double *A, int n, int r, int c, double v, int nm = nmin ) {
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
void setEntry( double *A, int n, int r, int c, double v, int nm=nmin ) {
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

void printMatrix( double *A, int n ) {
  for( int i = 0; i < n; i++ ) {
    for( int j = 0; j < n; j++ )
      printf("%.2f ", A[i+j*n]);
    printf("\n");
  }
}


int main( int argc, char **argv ) {
  int size = 256;
  // generate a symmetric, positive definite matrix
  double *M = (double *) malloc( size*size*sizeof(double) );
  fill( M, size*size);  
  double *Afull = (double *) malloc( size*size*sizeof(double) );
  char T = 'T', N = 'n';
  double one = 1., zero = 0.;
  dgemm_( &T, &N, &size, &size, &size, &one, M, &size, M, &size, &zero, Afull, &size );
  double *A = (double *) malloc( size*(size+1)/2*sizeof(double) );
  double *Acopy = (double *) malloc( size*(size+1)/2*sizeof(double) );
  for( int r = 0; r < size; r++ )
    for( int c = 0; c <= r; c++ ) {
      setEntry(A, size, r, c, Afull[r*size+c]);
      setEntry(Acopy, size, r, c, Afull[r*size+c], size);
    }
  //printMatrix(Afull,size);
  double startTime = read_timer();
  chol(A,size);
  double endTime = read_timer();
  printf("Time %f Gflop/s %f\n", endTime-startTime, size*size*size/3./(endTime-startTime)/1.e9); 
  int info = 0;
  char L = 'L';
  startTime = read_timer();
  dpptrf_( &L, &size, Acopy, &info);
  endTime = read_timer();
  printf("dpptrf Time %f Gflop/s %f\n", endTime-startTime, size*size*size/3./(endTime-startTime)/1.e9); 
  printf("info is %d, size is %d\n", info, size );
  startTime = read_timer();
  dpotrf_( &L, &size, Afull, &size, &info);
  endTime = read_timer();
  printf("dpotrf Time %f Gflop/s %f\n", endTime-startTime, size*size*size/3./(endTime-startTime)/1.e9); 
  //printMatrix(Afull,size);

  double maxDev = 0.;
  /*
  for( int r = 0; r < size; r++ )
    for( int c = 0; c <= r; c++ ) {
      maxDev = max(maxDev,fabs(Afull[r*size+c]-getEntry(A,size,r,c)));
      Afull[r*size+c] = getEntry(A,size,r,c);
    }
  */
  for( int r = 0; r < size; r++ )
    for( int c = 0; c < size; c++ ) {
      maxDev = max(maxDev,fabs(getEntry(Acopy,size,r,c,size)-getEntry(A,size,r,c)));
      //      Afull[r*size+c] = getEntry(Acopy,size,r,c,size);
    }
  //printMatrix(Afull,size);
  printf("Max deviation: %f\n", maxDev);
  //for( int r = 0; r < size; r++ )
  //  for( int c = 0; c < size; c++ ) {
  //    Afull[r*size+c] = getEntry(A,size,r,c);
  //  }
  //printMatrix(Afull,size);
  return 0;
}
