#include "types.h"
#include <mpi.h>

Matrix *gather( Matrix *m, int rank, int P ) {
  MPI_Datatype MPI_ENTRY;
  MPI_Type_contiguous( sizeof(Entry), MPI_CHAR, &MPI_ENTRY );
  MPI_Type_commit( &MPI_ENTRY );

  Matrix *ret = new Matrix(m->begin(), m->end());

  if( rank == 0 ) {
    for( int source = 1; source < P; source++ ) {
      int s;
      MPI_Recv( &s, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      Entry buf[s];
      MPI_Recv( buf, s, MPI_ENTRY, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      ret->insert( ret->end(), buf, (buf+s) );
    }
  } else {
    int_e s = m->size();
    MPI_Send( &s, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
    MPI_Send( m->data(), s, MPI_ENTRY, 0, 1, MPI_COMM_WORLD );
  }
  return ret;
}
