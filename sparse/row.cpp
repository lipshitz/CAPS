#include "types.h"
#include "tags.h"
#include "local-multiply.h"
#include "library.h"
#include <mpi.h>

Matrix *blockRow( Matrix *A, Matrix *B, int_d n, int rank, int P, vector<double> *times ) {

  double comptime = -read_timer();
  Matrix *C = local_multiply( A, B );
  comptime += read_timer();

  Entry *oldData = B->data();
  Entry *currentData;
  int_e oldSize = B->size();

  MPI_Datatype MPI_ENTRY;
  MPI_Type_contiguous( sizeof(Entry), MPI_CHAR, &MPI_ENTRY );
  MPI_Type_commit( &MPI_ENTRY );

  MPI_Datatype MPI_INT_E;
  MPI_Type_contiguous( sizeof(int_e), MPI_CHAR, &MPI_INT_E );
  MPI_Type_commit( &MPI_INT_E );

  int target = (rank+1)%P;
  int source = (rank+P-1)%P;
  double commtime;
  for( int i = 1; i < P; i++ ) {
    commtime -= read_timer();
    MPI_Request or1, or2;
    MPI_Isend( &oldSize, 1, MPI_INT_E, target, TAG_BR1, MPI_COMM_WORLD, &or1 );
    MPI_Isend( oldData, oldSize, MPI_ENTRY, target, TAG_BR2, MPI_COMM_WORLD, &or2 );
    int_e currentSize;
    MPI_Recv( &currentSize, 1, MPI_INT_E, source, TAG_BR1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    currentData = new Entry[currentSize];
    MPI_Recv( currentData, currentSize, MPI_ENTRY, source, TAG_BR2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

    double t = read_timer();
    commtime += t;
    comptime -= t;
    
    C = local_multiply( A->data(), A->size(), currentData, currentSize, C );

    t = read_timer();
    comptime += t;
    commtime -= t;

    MPI_Wait( &or1, MPI_STATUS_IGNORE );
    MPI_Wait( &or2, MPI_STATUS_IGNORE );
    commtime += read_timer();
    if( i > 1 )
      delete[] oldData;
    if( i == P-1 )
      delete[] currentData;
    oldSize = currentSize;
    oldData = currentData;
  }
  double sorttime = -read_timer();
  Matrix *ret = sortDedup( C, CompColMajorEntry );
  sorttime += read_timer();

  if( times ) {
    times->push_back( comptime );
    times->push_back( commtime );
    times->push_back( sorttime );
  }

  return ret;
}
