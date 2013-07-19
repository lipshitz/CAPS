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
  double *A = (double*) malloc( sizeSq*sizeof(double) );
  double *B = (double*) malloc( sizeSq*sizeof(double) );
  double *C = (double*) malloc( sizeSq*sizeof(double) );
  srand48(getRank());
  fill(A,sizeSq);
  fill(B,sizeSq);
  fill(C,sizeSq);
  if( getRank() == 0 )
    printf("Starting benchmark\n");
  MPI_Barrier( MPI_COMM_WORLD );
  double startTime = read_timer();
  mult( A, B, C, size, P, r );
  MPI_Barrier( MPI_COMM_WORLD );
  double endTime = read_timer();
  
  if( getRank() == 0 )
    printf("Time: %f Gflop/s %f\n", endTime-startTime, size*1.*size*size*2./(endTime-startTime)/1.e9);

  printCounters(size);
  free(A);
  free(B);
  free(C);
  MPI_Finalize();
}
