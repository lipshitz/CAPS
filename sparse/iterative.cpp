#include "types.h"
#include "local-multiply.h"
#include "merge.h"
#include "library.h"
#include <mpi.h>

Matrix *iterative3D( Matrix *A, Matrix *B, int_d n, int c, int sqrtPoc, int sqrtc, MPI_Comm rowComm, int rowRank, MPI_Comm colComm, int colRank, MPI_Comm fibComm, int fibRank, vector<double> *times ) {
  MPI_Datatype MPI_INT_E;
  MPI_Type_contiguous( sizeof(int_e), MPI_CHAR, &MPI_INT_E );
  MPI_Type_commit( &MPI_INT_E );

  MPI_Datatype MPI_ENTRY;
  MPI_Type_contiguous( sizeof(Entry), MPI_CHAR, &MPI_ENTRY );
  MPI_Type_commit( &MPI_ENTRY );

  int_d lbs = n/sqrtPoc; // this is the size of a block shared by a fiber
  int_d firstCol = lbs*rowRank;
  int_d firstRow = lbs*colRank;
  int_d colsPerProc = lbs/c;
  // All-to-all along fibers to redistribute A
  int_e sendCounts[c];
  for( int j = 0; j < c ; j++ )
    sendCounts[j] = 0;
  int i = 0;
  double time_shuffle = -read_timer();
  for( auto it = A->begin(); it != A->end(); it++ ) {
    if( it->first.second < firstCol+(i+1)*colsPerProc )
      sendCounts[i]++;
    else {
      i++;
      it--;
    }
  }
  time_shuffle += read_timer();
  double time_comm = -read_timer();
  int_e recvCounts[c];
  MPI_Alltoall( sendCounts, 1, MPI_INT_E, recvCounts, 1, MPI_INT_E, fibComm );
  int_e sendOffsets[c];
  sendOffsets[0] = 0;
  for( int i = 1 ; i < c; i++ )
    sendOffsets[i] = sendOffsets[i-1]+sendCounts[i-1];
  int_e recvOffsets[c];
  recvOffsets[0] = 0;
  for( int i = 1 ; i < c; i++ )
    recvOffsets[i] = recvOffsets[i-1]+recvCounts[i-1];
  int_e recvTotal = recvOffsets[c-1] + recvCounts[c-1];
  Entry *buf = new Entry[recvTotal];
  MPI_Alltoallv( A->data(), sendCounts, sendOffsets, MPI_ENTRY, buf, recvCounts, recvOffsets, MPI_ENTRY, fibComm );
  time_comm += read_timer();
  time_shuffle -= read_timer();
  // sort buf, since it won't be in order anymore
  Matrix *A2 = merge( buf, recvCounts, recvOffsets, c, ReverseColMajor );
  delete[] buf;
    
  // All-to-all along fibers to redistribute B 
  int_d rowsPerProc = colsPerProc;
  for( int j = 0; j < c ; j++ )
    sendCounts[j] = 0;
  i = 0;
  for( auto it = B->begin(); it != B->end(); it++ ) {
    if( it->first.first < firstRow+(i+1)*rowsPerProc )
      sendCounts[i]++;
    else {
      i++;
      it--;
    }
  }
  time_shuffle += read_timer();
  time_comm -= read_timer();
  MPI_Alltoall( sendCounts, 1, MPI_INT_E, recvCounts, 1, MPI_INT_E, fibComm );
  sendOffsets[0] = 0;
  for( int i = 1 ; i < c; i++ )
    sendOffsets[i] = sendOffsets[i-1]+sendCounts[i-1];
  recvOffsets[0] = 0;
  for( int i = 1 ; i < c; i++ )
    recvOffsets[i] = recvOffsets[i-1]+recvCounts[i-1];
  recvTotal = recvOffsets[c-1] + recvCounts[c-1];
  buf = new Entry[recvTotal];
  MPI_Alltoallv( B->data(), sendCounts, sendOffsets, MPI_ENTRY, buf, recvCounts, recvOffsets, MPI_ENTRY, fibComm );
  time_comm += read_timer();
  time_shuffle -= read_timer();
  // sort buf, since it won't be in order anymore
  Matrix *B2 = merge( buf, recvCounts, recvOffsets, c, ReverseRowMajor );
  delete[] buf;
  time_shuffle += read_timer();

  time_comm -= read_timer();
  Entry *A3, *B3;
  int_e sA, sB;
  // All-gather A along rows
  int_e sendSize = A2->size();
  int_e recvSizes[sqrtPoc];
  MPI_Allgather( &sendSize, 1, MPI_INT_E, recvSizes, 1, MPI_INT_E, rowComm );
  int_e recvOffsets2[sqrtPoc];
  recvOffsets2[0] = 0;
  for( int j = 1; j < sqrtPoc; j++ )
    recvOffsets2[j] = recvOffsets2[j-1]+recvSizes[j-1];
  sA = recvOffsets2[sqrtPoc-1]+recvSizes[sqrtPoc-1];
  A3 = new Entry[sA];
  MPI_Allgatherv( A2->data(), sendSize, MPI_ENTRY, A3, recvSizes, recvOffsets2, MPI_ENTRY, rowComm );
  delete A2;
  // A3 is automatically sorted correctly
  
  // All-gather B along columns
  sendSize = B2->size();
  MPI_Allgather( &sendSize, 1, MPI_INT_E, recvSizes, 1, MPI_INT_E, colComm );
  recvOffsets2[0] = 0;
  for( int j = 1; j < sqrtPoc; j++ )
    recvOffsets2[j] = recvOffsets2[j-1]+recvSizes[j-1];
  sB = recvOffsets2[sqrtPoc-1]+recvSizes[sqrtPoc-1];
  B3 = new Entry[sB];
  MPI_Allgatherv( B2->data(), sendSize, MPI_ENTRY, B3, recvSizes, recvOffsets2, MPI_ENTRY, colComm );
  delete B2;
  // B3 is automatically sorted correctly
  time_comm += read_timer();
  double time_comp = -read_timer();
  // local multiplication
  Matrix *loc = local_multiply( A3, sA, B3, sB );
  time_comp += read_timer();
  delete[] A3;
  delete[] B3;
  int_d bs = lbs/sqrtc;
  double denom = 1.*sqrtc/lbs;
  Matrix *local = sortDedup( loc, ([=](const Entry c1, const Entry c2) -> bool {
	int_d j1 = c1.first.second * denom;
	int_d j2 = c2.first.second * denom;
	//int j1 = c1.first.second / bs;
	//int j2 = c2.first.second / bs;
	if( j1 < j2 ) return true;
	if( j1 > j2 ) return false;
	int_d i1 = c1.first.first * denom;
	int_d i2 = c2.first.first * denom;
	//int i1 = c1.first.first / bs;
	//int i2 = c2.first.first / bs;
	if( i1 < i2 ) return true;
	if( i1 > i2 ) return false;
	return CompColMajorEntry(c1,c2);
      }), true, times );
  
  // C is sorted by blocks, just need to mark the boundaries
  time_shuffle -= read_timer();
  for( int k = 0; k < c; k++ )
    sendCounts[k] = 0;
  i = 0;
  int j=0;
  for( auto it = local->begin(); it != local->end(); it++ ) {
    if( it->first.first < firstRow+(i+1)*bs && it->first.second < firstCol+(j+1)*bs )
      sendCounts[i+j*sqrtc]++;
    else if( i < sqrtc-1 ) {
      i++;
      it--;
    } else {
      j++;
      i = 0;
      if( j >= sqrtc ) {
	exit(-1);}
      it--;
    }
  }
  time_shuffle += read_timer();
  time_comm -= read_timer();
  MPI_Alltoall( sendCounts, 1, MPI_INT_E, recvCounts, 1, MPI_INT_E, fibComm );
  sendOffsets[0] = 0;
  for( int i = 1 ; i < c; i++ )
    sendOffsets[i] = sendOffsets[i-1]+sendCounts[i-1];
  recvOffsets[0] = 0;
  for( int i = 1 ; i < c; i++ )
    recvOffsets[i] = recvOffsets[i-1]+recvCounts[i-1];
  recvTotal = recvOffsets[c-1] + recvCounts[c-1];
  buf = new Entry[recvTotal];
  MPI_Alltoallv( local->data(), sendCounts, sendOffsets, MPI_ENTRY, buf, recvCounts, recvOffsets, MPI_ENTRY, fibComm );
  // sort buf, since it won't be in order anymore
  time_comm += read_timer();
  time_shuffle -= read_timer();
  Matrix *C = merge( buf, recvCounts, recvOffsets, c, ReverseColMajor );
  time_shuffle += read_timer();
  delete[] buf;

  if( times ) {
    times->push_back( time_shuffle );
    times->push_back( time_comp );
    times->push_back( time_comm );
  }

  return C;
}
