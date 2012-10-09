#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const int TAG_PROGRAM_EXIT = 100;

int main( int argc, char**argv ) {
  MPI_Init(&argc, &argv);
  int nProcs, rank;
  MPI_Comm_size( MPI_COMM_WORLD, &nProcs );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // Check that nProcs is a power of 7
  int i = nProcs;
  int log7nProcs = 0;
  while( i > 1 ) {
    i /= 7;
    log7nProcs++;
  }
  if( log7nProcs == 0 || i != 1 ) {
    if( rank == 0 )
      printf("Run with a nonzero power of 7 processors\n");
    MPI_Finalize();
    exit(-1);
  }
  MPI_Group initialGroup;
  MPI_Comm_group( MPI_COMM_WORLD, &initialGroup );

  MPI_Group groupAtLevel[log7nProcs];
  for( int i = 0, procsThisLevel = nProcs; i < log7nProcs; i++, procsThisLevel /= 7 ) {
    int firstProc = (rank/procsThisLevel)*procsThisLevel;
    int ranks[procsThisLevel];
    for( int j = 0; j < procsThisLevel; j++ )
      ranks[j] = firstProc+j;
    MPI_Group_incl( initialGroup, procsThisLevel, ranks, groupAtLevel+i );
  }
  MPI_Comm commAtLevel[log7nProcs];
  int rankAtLevel[log7nProcs];
  for( int i = 0; i < log7nProcs; i++ ) {
    MPI_Comm_create( MPI_COMM_WORLD, groupAtLevel[i], commAtLevel+i );
    MPI_Group_rank( groupAtLevel[i], rankAtLevel+i );
  }

  printf("%d %d\n", rankAtLevel[0], rankAtLevel[1]);


  MPI_Finalize();
  return 0;
}
