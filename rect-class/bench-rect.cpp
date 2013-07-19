#include "communication.h"
#include "rectmm.h"
#include "rectsizes.h"
#include "library.h"

void fillInt( double *A, int n ) {
  for( int i = 0; i < n; i++ )
    A[i] = double((int)( 10*drand48()));
}

int main( int argc, char **argv ) {
  initCommunication(&argc, &argv);

  int rank = getRank();

  double *A, *B, *C;

  int m = read_int( argc, argv, "-m", 128 );
  int n = read_int( argc, argv, "-n", 128 );
  int k = read_int( argc, argv, "-k", 128 );
  int r = read_int( argc, argv, "-r", 0 );

  char *pattern = read_string( argc, argv, "-p", NULL );
  if( r > 0 && !pattern ) {
    printf( "pattern required, specify with -p\n" );
    MPI_Finalize();
    exit(-1);
  }
  char *divPatternString = read_string( argc, argv, "-d", NULL );
  int divPattern[3*r];
  if( r > 0 && !divPattern ) {
    printf( "division pattern required, specify with -d\n" );
    MPI_Finalize();
    exit(-1);
  } else {
    for( int i = 0; i < 3*r; i++ ) {
      char t[5];
      sprintf(t, "%c", divPatternString[i]);
      divPattern[i] = atoi(t);
      if( divPattern[i] == 0 ) {
	printf("Illegal character in division pattern: %s\n", t);
      }
    }
  }
  int P;
  MPI_Comm_size( MPI_COMM_WORLD, &P );

  if( getRank() == 0 ) {
    printf("Benchmarking %dx%dx%d multiplication using %d processes\n", m,n,k,P);
    printf("Division pattern: ");
    for( int i = 0; i < r; i++ ) {
      for( int j = 0; j < 3; j++ )
	printf("%d,",divPattern[3*i+j]); 
      printf(" ");
    }
    printf("execution pattern: %s\n", pattern);
  }

  initSizesRect( m, n, k, P, r, divPattern );

  // allocate the initial matrices
  A = (double*) malloc( sizeof(double)*getSizeRect(m,1,k,P) );
  B = (double*) malloc( sizeof(double)*getSizeRect(1,n,k,P) );
  C = (double*) malloc( sizeof(double)*getSizeRect(m,n,1,P) );

  // fill the matrices with random data
  srand48(getRank());
  fillInt( A, getSizeRect(m,1,k,P) );
  fillInt( B, getSizeRect(1,n,k,P) );


  MPI_Barrier( MPI_COMM_WORLD );
  double startTime = read_timer();
  rectMM( A, B, C, m, n, k, P, r, pattern, divPattern );
  MPI_Barrier( MPI_COMM_WORLD );
  double endTime = read_timer();

  if( getRank() == 0 )
    printf("Benchmark took %f seconds so %f Gflop/s\n", endTime-startTime, 2.*m*n*k/1.e9/(endTime-startTime) );

  for( int i = 0; i < P; i++ ) {
    if( getRank() == i ) {
      printf("(%d) A: ", i );
      for( int j = 0; j < getSizeRect(m,1,k,P); j++ )
	printf("%5.3f ", A[j] );
      printf("\n");
      printf("(%d) B: ", i );
      for( int j = 0; j < getSizeRect(1,n,k,P); j++ )
	printf("%5.3f ", B[j] );
      printf("\n");
      printf("(%d) C: ", i );
      for( int j = 0; j < getSizeRect(m,n,1,P); j++ )
	printf("%5.3f ", C[j] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }

  free(A);
  free(B);
  free(C);

  MPI_Finalize();
}
