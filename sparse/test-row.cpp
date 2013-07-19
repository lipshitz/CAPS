#include "row.h"
#include "testlib.h"
#include "local-multiply.h"
#include "types.h"
#include "generate.h"
#include <mpi.h>

int main( int argc, char **argv ) {
  MPI_Init( &argc, &argv );
  int rank, P;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &P );

  int s = read_int( argc, argv, "-s", 5 );
  int d = read_int( argc, argv, "-d", 1 );
  int ds = read_int( argc, argv, "-ds", 0 );
  int_d n = ((int_d) 1) << s;
  int_d colsPerProc = n/P;
  n = colsPerProc * P;
  double density = 1.*d/n/(1 << ds);

  BlockIndexColMajor ai = BlockIndexColMajor(colsPerProc*rank, 0, colsPerProc, n );
  BlockIndexRowMajor bi = BlockIndexRowMajor(colsPerProc*rank, 0, colsPerProc, n );

  //printf("%d Generating matrices\n", rank);
  Matrix *A = generateMatrix( &ai, density, time(0)+100*rank );
  Matrix *B = generateMatrix( &bi, density, time(0)+100*rank+1 );
  //Matrix *A = generateMatrix( &ai, density, 2*rank+39 );
  //Matrix *B = generateMatrix( &bi, density, 2*rank+1+39 );

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
  Matrix *C = blockRow( A, B, n, rank, P );

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
