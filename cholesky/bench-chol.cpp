#include "chol.h"
#include "syrk.h"
#include "communication.h"
#include "library.h"
#include "sizes.h"
#include <stdio.h>
#include <math.h>
#include "counters.h"

int main( int argc, char **argv ) {

  initCommunication( &argc, &argv );
  
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
  int sizeSq = getSizeSq(r,P);
  int sizeTri = getSizeTri(r,P);
  double *X = (double*) malloc( sizeSq*sizeof(double) );
  srand48(getRank());
  fill(X,sizeSq);
  double *A = (double*) malloc( sizeTri*sizeof(double) );
  if( getRank() == 0 )
    printf("Generating a symmetric positive definite test matrix\n");
  initTimers();
  MPI_Barrier( MPI_COMM_WORLD );
  double st2 = read_timer();
  syrk( A, X, size, P, r, 0. );
  MPI_Barrier( MPI_COMM_WORLD );
  double et2 = read_timer();
  if( getRank() == 0 )
    printf("Generation time: %f\n", et2-st2);
  initTimers();
  free(X);
  for( int i = 0; i < sizeTri; i++ )
    A[i] = -A[i];

  if( getRank() == 0 )
    printf("Starting benchmark\n");
  MPI_Barrier( MPI_COMM_WORLD );
  double startTime = read_timer();
  chol( A, size, P, r );
  MPI_Barrier( MPI_COMM_WORLD );
  double endTime = read_timer();
  
  if( getRank() == 0 )
    printf("Time: %f Gflop/s %f\n", endTime-startTime, size*1.*size*size/3./(endTime-startTime)/1.e9);

  free(A);
  printCounters(size);
  MPI_Finalize();
}
