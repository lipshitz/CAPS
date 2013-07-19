#include "summa.h"
#include "testlib.h"
#include "local-multiply.h"
#include "types.h"
#include "generate.h"
#include "library.h"
#include <mpi.h>

int main( int argc, char **argv ) {
  MPI_Init( &argc, &argv );
  int rank, P;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &P );

  int s = read_int( argc, argv, "-s", 10 );
  int d = read_int( argc, argv, "-d", 1 );
  int ds = read_int( argc, argv, "-ds", 0 );
  int_d n = ((int_d) 1) << s;
  //  int_d n = read_int( argc, argv, "-n", 1<<s );
  int v = read_int( argc, argv, "-v", 0 );

  int sqrtP = (int) sqrt(1.*P);
  if( sqrtP*sqrtP != P ) {
    if( rank == 0 )
      printf("Requires a square processor grid\n");
    MPI_Finalize();
    exit(-1);
  }

  // construct row and column communicators                                                                                
  MPI_Group initialGroup;
  MPI_Comm_group( MPI_COMM_WORLD, &initialGroup );
  MPI_Comm rowComm;
  MPI_Comm colComm;
  MPI_Group gp;
  MPI_Comm cm;
  for( int i = 0; i < sqrtP; i++ ) {
    int rranks[sqrtP];
    int cranks[sqrtP];
    bool rc = false, cc = false;
    for( int j = 0; j < sqrtP; j++ ) {
      rranks[j] = sqrtP*i+j;
      cranks[j] = sqrtP*j+i;
      if( rranks[j] == rank )
        rc = true;
      if( cranks[j] == rank )
        cc = true;
    }
    MPI_Group_incl( initialGroup, sqrtP, rranks, &gp );
    MPI_Comm_create( MPI_COMM_WORLD, gp, &cm );
    if( rc )
      rowComm = cm;
    MPI_Group_free( &gp );
    MPI_Group_incl( initialGroup, sqrtP, cranks, &gp );
    MPI_Comm_create( MPI_COMM_WORLD, gp, &cm );
    if( cc )
      colComm = cm;
    MPI_Group_free( &gp );
  }
  MPI_Group_free( &initialGroup );

  int rrank, crank;
  MPI_Comm_rank( rowComm, &rrank );
  MPI_Comm_rank( colComm, &crank );

  int_d bs = (n+sqrtP-1)/sqrtP;
  int_d bscThisProc = min(bs,n-crank*bs);
  int_d bsrThisProc = min(bs,n-crank*bs);
  double density = 1.*d/n/(1<<ds);

  BlockIndexColMajor ai = BlockIndexColMajor(crank*bs, rrank*bs, bsrThisProc, bscThisProc );
  BlockIndexRowMajor bi = BlockIndexRowMajor(crank*bs, rrank*bs, bsrThisProc, bscThisProc );

  Matrix *A = generateMatrix( &ai, density, time(0)+100*rank );
  Matrix *B = generateMatrix( &bi, density, time(0)+100*rank+50 );
  Matrix *C;
  vector<double> times;
  MPI_Barrier( MPI_COMM_WORLD );
  double startTime = read_timer();
  if( v ) {
    C = spSUMMA( A, B, n, sqrtP, rowComm, rrank, colComm, crank, &times );    
  } else {
    C = spSUMMA( A, B, n, sqrtP, rowComm, rrank, colComm, crank );
  }
  MPI_Barrier( MPI_COMM_WORLD );
  double stopTime = read_timer();
  if( rank == 0 ) {
    printf("n %ld d %d/(2^%d) P %d time %f matrix sizes this proc %lu %lu %lu\n", n, d, ds, P, stopTime-startTime, A->size(), B->size(), C->size());
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
