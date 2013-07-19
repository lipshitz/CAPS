#include "iterative.h"
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
  //int_d n = read_int( argc, argv, "-n", 1<<s );
  int v = read_int( argc, argv, "-v", 0 );
  int c = read_int( argc, argv, "-c", 1 );

  int sqrtP = (int) sqrt(1.*P);
  if( sqrtP*sqrtP != P ) {
    if( rank == 0 )
      printf("Requires a square processor grid\n");
    MPI_Finalize();
    exit(-1);
  }
  int sqrtc = (int) sqrt(1.*c);
  if( sqrtc*sqrtc != c ) {
    if( rank == 0 )
      printf("Requires a square value of c\n");
    MPI_Finalize();
    exit(-1);
  }
  int sqrtPoc = sqrtP/sqrtc;
  if( sqrtPoc*sqrtc != sqrtP ) {
    if( rank == 0 )
      printf("Requires sqrt(c) to divide sqrt(P)\n");
    MPI_Finalize();
    exit(-1);
  }

  // construct row, column, and fiber communicators                                                                                    
  MPI_Group initialGroup;
  MPI_Comm_group( MPI_COMM_WORLD, &initialGroup );
  MPI_Comm rowComm;
  MPI_Comm colComm;
  MPI_Comm fibComm;
  MPI_Group gp;
  MPI_Comm cm;
  for( int i = 0; i < sqrtPoc; i++ ) {
    for( int k = 0; k < c; k++ ) {
      int rranks[sqrtPoc];
      int cranks[sqrtPoc];
      bool rc = false, cc = false;
      for( int j = 0; j < sqrtPoc; j++ ) {
        cranks[j] = (sqrtPoc*i+j)*c+k;
        rranks[j] = (sqrtPoc*j+i)*c+k;
        if( rranks[j] == rank )
          rc = true;
        if( cranks[j] == rank )
          cc = true;
      }
      MPI_Group_incl( initialGroup, sqrtPoc, rranks, &gp );
      MPI_Comm_create( MPI_COMM_WORLD, gp, &cm );
      if( rc )
        rowComm = cm;
      MPI_Group_free( &gp );
      MPI_Group_incl( initialGroup, sqrtPoc, cranks, &gp );
      MPI_Comm_create( MPI_COMM_WORLD, gp, &cm );
      if( cc )
        colComm = cm;
      MPI_Group_free( &gp );
    }
  }
  for( int i = 0; i < sqrtPoc; i++ ) {
    for( int j = 0; j < sqrtPoc; j++ ) {
      int franks[c];
      bool fc = false;
      for( int k = 0; k < c; k++ ) {
        franks[k] = k+(sqrtPoc*i+j)*c;
        if( franks[k] == rank )
          fc = true;
      }
      MPI_Group_incl( initialGroup, c, franks, &gp );
      MPI_Comm_create( MPI_COMM_WORLD, gp, &cm );
      if( fc )
        fibComm = cm;
      MPI_Group_free( &gp );
    }
  }
  MPI_Group_free( &initialGroup );

  int rrank, crank, frank;
  MPI_Comm_rank( rowComm, &rrank );
  MPI_Comm_rank( colComm, &crank );
  MPI_Comm_rank( fibComm, &frank );

  int_d bs = (n+sqrtP-1)/sqrtP;
  int_d bscThisProc = min(bs,n-crank*bs);
  int_d bsrThisProc = min(bs,n-crank*bs);
  double density = 1.*d/n/(1<<ds);
  int bc = rrank*sqrtc+frank/sqrtc;
  int br = crank*sqrtc+frank%sqrtc;

  BlockIndexColMajor ai = BlockIndexColMajor(br*bs, bc*bs, bsrThisProc, bscThisProc );
  BlockIndexRowMajor bi = BlockIndexRowMajor(br*bs, bc*bs, bsrThisProc, bscThisProc );

  Matrix *A = generateMatrix( &ai, density, time(0)+100*rank );
  Matrix *B = generateMatrix( &bi, density, time(0)+100*rank+50 );
  Matrix *C;
  vector<double> times;
  MPI_Barrier( MPI_COMM_WORLD );
  double startTime = read_timer();
  if( v ) {
    C = iterative3D( A, B, n, c, sqrtPoc, sqrtc, rowComm, rrank, colComm, crank, fibComm, frank, &times );
  } else {
    C = iterative3D( A, B, n, c, sqrtPoc, sqrtc, rowComm, rrank, colComm, crank, fibComm, frank );
  }
  MPI_Barrier( MPI_COMM_WORLD );
  double stopTime = read_timer();
  if( rank == 0 ) {
    printf("n %ld d %d/(2^%d) P %d c %d time %f matrix sizes this proc %lu %lu %lu\n", n, d, ds, P, c, stopTime-startTime, A->size(), B->size(), C->size());
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
