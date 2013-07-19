#include "local-multiply.h"
#include "library.h"
#include "merge.h"
#include <mpi.h>
#include <functional>
#include <algorithm>

Matrix *outerProduct( Matrix *A, Matrix *B, int_d n, int P, vector<double> *times ) {
  double t1 = read_timer();
  Matrix *loc = local_multiply(A,B);
  double t2 = read_timer();
  Matrix *local = sortDedup( loc, CompColMajorEntry, true, times );
  // stream through the answer and determine how much goes where
  int_d colsPerProc = (n+P-1)/P;
  int sendCounts[P];
  for( int i = 0; i < P; i++ )
    sendCounts[i] = 0;
  int i = 0;
  for( auto it = local->begin(); it != local->end(); it++ ) {
    if( it->first.second < (i+1)*colsPerProc )
      sendCounts[i]++;
    else {
      i++;
      it--;
    }
  }
  double t3 = read_timer();

  int recvCounts[P];

  MPI_Alltoall( sendCounts, 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD );
  double t4 = read_timer();

  int sendOffsets[P];
  sendOffsets[0] = 0;
  for( int i = 1 ; i < P; i++ )
    sendOffsets[i] = sendOffsets[i-1]+sendCounts[i-1];
  int recvOffsets[P];
  recvOffsets[0] = 0;
  for( int i = 1 ; i < P; i++ )
    recvOffsets[i] = recvOffsets[i-1]+recvCounts[i-1];
  int recvTotal = recvOffsets[P-1] + recvCounts[P-1];
  Entry *buf = new Entry[recvTotal];

  MPI_Datatype MPI_ENTRY;
  MPI_Type_contiguous( sizeof(Entry), MPI_CHAR, &MPI_ENTRY );
  MPI_Type_commit( &MPI_ENTRY );
  
  double t5 = read_timer();

  MPI_Alltoallv( local->data(), sendCounts, sendOffsets, MPI_ENTRY, buf, recvCounts, recvOffsets, MPI_ENTRY, MPI_COMM_WORLD );

  double t6 = read_timer();

  delete local;

  Matrix *ret = merge( buf, recvCounts, recvOffsets, P, ReverseColMajor );

  delete[] buf;

  double t7 = read_timer();

  if( times ) {
    times->push_back(t2-t1); // compute time
    times->push_back(t4-t3+t6-t5); // communicate time
    times->push_back(t3-t2+t5-t4+t7-t6); // sort time, etc
  }

  return ret;
}
