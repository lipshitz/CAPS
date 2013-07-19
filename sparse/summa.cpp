#include "types.h"
#include "local-multiply.h"
#include "library.h"
#include <mpi.h>

Matrix *spSUMMA( Matrix *A, Matrix *B, int_d n, int sqrtP, MPI_Comm rowComm, int rowRank, MPI_Comm colComm, int colRank, vector<double> *times ) {
  MPI_Datatype MPI_ENTRY;
  MPI_Type_contiguous( sizeof(Entry), MPI_CHAR, &MPI_ENTRY );
  MPI_Type_commit( &MPI_ENTRY );

  MPI_Datatype MPI_INT_E;
  MPI_Type_contiguous( sizeof(int_e), MPI_CHAR, &MPI_INT_E );
  MPI_Type_commit( &MPI_INT_E );

  Matrix *C = new Matrix;
  Entry *Abuf, *Bbuf;
  int_e sA, sB;
  double comm = 0., comp = 0.;
  for( int k = 0; k < sqrtP; k++ ) {
    comm -= read_timer();
    // broadcast A along rows
    if( rowRank == k ) {
      sA = A->size();
      Abuf = A->data();
    }
    MPI_Bcast( &sA, 1, MPI_INT_E, k, rowComm );
    if( rowRank != k )
      Abuf = new Entry[sA];
    MPI_Bcast( Abuf, sA, MPI_ENTRY, k, rowComm );

    // broadcast B along cols
    if( colRank == k ) {
      sB = B->size();
      Bbuf = B->data();
    }
    MPI_Bcast( &sB, 1, MPI_INT_E, k, colComm );
    if( colRank != k )
      Bbuf = new Entry[sB];
    MPI_Bcast( Bbuf, sB, MPI_ENTRY, k, colComm );
    comm += read_timer();

    comp -= read_timer();
    C = local_multiply( Abuf, sA, Bbuf, sB, C );
    comp += read_timer();

    if( rowRank != k )
      delete[] Abuf;
    if( colRank != k )
      delete[] Bbuf;

  }
  Matrix *ret = sortDedup( C, CompColMajorEntry, true, times );
  if( times ) {
    times->push_back( comp );
    times->push_back( comm );
  }
  return ret;
}
