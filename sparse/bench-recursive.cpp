#include "recursive.h"
#include "library.h"
#include "local-multiply.h"
#include "types.h"
#include "generate.h"
#include <mpi.h>
#include <cmath>

int main( int argc, char **argv ) {
  MPI_Init( &argc, &argv );
  int rank, P;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &P );

  int s = read_int( argc, argv, "-s", 10 );
  int d = read_int( argc, argv, "-d", 1 );
  int ds = read_int( argc, argv, "-ds", 0 );
  int_d n = ((int_d) 1) << s;
  int v = read_int( argc, argv, "-v", 0 );
  char *pattern = read_string( argc, argv, "-p", NULL );

  double density = 1.*d/n/(1 << ds);

  int Pc = P, logP = 0;
  while( Pc > 1 ) {
    logP += 1;
    Pc /= 2;
  }
  
  int patt[logP/2];
  if( pattern ) {
    for( int i = 0; i <= logP/2; i++ ) {
      if( pattern[i] == '0' )
	patt[i] = 0;
      else
	patt[i] = 1;
    }
  } else {
    for( int i = 0; i < logP/2 && (1 << i) < d; i++ )
      patt[i] = 0;
    for( int i = d; i < logP/2; i++ )
      patt[i] = 1;
  }

  MPI_Comm *comms = initCommunication( logP/2, patt );
  int blockrank = 0;
  for( int i = logP/2-1; i >= 0; i-- ) {
    int rrank;
    MPI_Comm_rank( comms[i], &rrank );
    blockrank += rrank * (1<<(2*(logP/2-1-i)));
  }

  int_d colsPerProc = n/P;
  BlockIndexRowMajor bi = BlockIndexRowMajor(colsPerProc*blockrank, 0, colsPerProc, n );
  BlockIndexColMajor ai = BlockIndexColMajor(0, colsPerProc*blockrank, n, colsPerProc );

  Matrix *A = generateMatrix( &ai, density, time(0)+100*rank );
  Matrix *B = generateMatrix( &bi, density, time(0)+100*rank+50 );
  Matrix *C;

  vector<double> times;
  MPI_Barrier( MPI_COMM_WORLD );
  double startTime = read_timer();
  if( v ) {
    C = recursiveMultiply( A, B, s, &times );
  } else {
    C = recursiveMultiply( A, B, s );
  }
  MPI_Barrier( MPI_COMM_WORLD );
  double stopTime = read_timer();
  if( rank == 0 ) {
    printf("n %ld d %d/(2^%d) P %d pattern ", n, d, ds, P );
    for( int i = 0; i < logP/2; i++ )
      printf("%d", patt[i]);
    printf(" time %f matrix sizes this proc %lu %lu %lu\n", stopTime-startTime, A->size(), B->size(), C->size());
  }
  if( v ) {
    double totalTimes[times.size()];
    double minTimes[times.size()];
    double maxTimes[times.size()];
    MPI_Reduce( times.data(), totalTimes, times.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
    MPI_Reduce( times.data(), minTimes, times.size(), MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD );
    MPI_Reduce( times.data(), maxTimes, times.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    if( rank == 0 ) {
      for( unsigned i = 0; i < times.size(); i++ )
        printf("%f (%f-%f)\n", totalTimes[i]/P,minTimes[i],maxTimes[i]);
    }
  }


  MPI_Finalize();
}
