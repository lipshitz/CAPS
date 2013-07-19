#include "iterative.h"
#include "testlib.h"
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

  int s = read_int( argc, argv, "-s", 5 );
  int d = read_int( argc, argv, "-d", 1 );
  int ds = read_int( argc, argv, "-ds", 0 );
  int c = read_int( argc, argv, "-c", P );
  int_d n = ((int_d) 1) << s;
  int_d colsPerProc = n/P;
  n = colsPerProc * P;
  double density = 1.*d/n/(1 << ds);
  int sqrtc = (int) (sqrt(1.*c));
  c = sqrtc * sqrtc;

  if( rank == 0 )
    printf("Using n=%ld, d=%d on P=%d processors, c=%d\n", n, d, P, c );

  int sqrtP = (int) sqrt(1.*P);
  if( sqrtP*sqrtP != P ) {
    if( rank == 0 )
      printf("Requires a square processor grid\n");
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
  int_d bs = n/sqrtP;

  int bc = rrank*sqrtc+frank/sqrtc;
  int br = crank*sqrtc+frank%sqrtc;

  BlockIndexColMajor ai = BlockIndexColMajor(br*bs, bc*bs, bs, bs );
  BlockIndexRowMajor bi = BlockIndexRowMajor(br*bs, bc*bs, bs, bs );

  //printf("%d Generating matrices\n", rank);
  Matrix *A = generateMatrix( &ai, density, time(0)+100*rank );
  Matrix *B = generateMatrix( &bi, density, time(0)+100*rank+50 );
  //Matrix *A = generateMatrix( &ai, density, 2*rank+29 );
  //Matrix *B = generateMatrix( &bi, density, 2*rank+1+29 );

  /*
  printf("A %d: ", rank);
  for( auto it = A->begin(); it != A->end(); it++ )
    printf("(%d %d) ", it->first.first, it->first.second);
  printf("\n");

  printf("B %d: ", rank);
  for( auto it = B->begin(); it != B->end(); it++ )
    printf("(%d %d) ", it->first.first, it->first.second);
  printf("\n");
  */

  //printf("%d collecting input\n", rank );
  Matrix *fullA = gather(A, rank, P );
  Matrix *fullB = gather(B, rank, P );

  //printf("%d dist multiply\n", rank );
  Matrix *C = iterative3D( A, B, n, c, sqrtPoc, sqrtc, rowComm, rrank, colComm, crank, fibComm, frank );

  /*
  printf("C %d: ", rank);
  for( auto it = C->begin(); it != C->end(); it++ )
    printf("(%d %d) ", it->first.first, it->first.second);
  printf("\n");
  */

  //printf("%d collecting output\n", rank );
  Matrix *fullC = gather(C, rank, P );

  /*
  printf("fC %d: ", rank);
  for( auto it = fullC->begin(); it != fullC->end(); it++ )
    printf("(%d %d) ", it->first.first, it->first.second);
  printf("\n");
  */

  //printf("%d checking answer\n", rank );
  if( rank == 0 ) {
    double maxError = 0.;
    std::sort( fullA->begin(), fullA->end(), CompColMajorEntry );
    std::sort( fullB->begin(), fullB->end(), CompRowMajorEntry );
    Matrix *testC = sortDedup( local_multiply( fullA, fullB ), CompColMajorEntry );
    printf("sizes %lu vs %lu\n", testC->size(), fullC->size());
    std::sort( fullC->begin(), fullC->end(), CompColMajorEntry );
    for( auto it1 = testC->begin(), it2 = fullC->begin();
	 it1 != testC->end() && it2 != fullC->end();
	 it1++, it2++ ) {
      if( it1->first.first != it2->first.first ||
	  it1->first.second != it2->first.second )
	printf("Error: %ld %ld vs %ld %ld\n", it1->first.first, it1->first.second, it2->first.first, it2->first.second );
      else {
	if( it2->second-it1->second > maxError )
	  maxError = it2->second-it1->second;
	if( it1->second-it2->second > maxError )
	  maxError = it1->second-it2->second;
      }
    }
    printf("max error %e\n", maxError );
  }
  MPI_Finalize();
}
