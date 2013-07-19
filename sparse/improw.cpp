#include "types.h"
#include "tags.h"
#include "local-multiply.h"
#include "library.h"
#include <mpi.h>

bool ReverseRowMajorPair( pair<int,int_d> p1, pair<int,int_d> p2 ) {
  return (p1.second > p2.second);
}

Matrix *blockRow( Matrix *A, Matrix *B, int_d n, int rank, int P, vector<double> *times ) {

  double tfind = -read_timer();
  // find all the rows of B that we need, indexed by processor
  vector<int_d> rowsNeeded[P];
  int_d last_req = -1;
  int_d block_size = (n+P-1)/P;
  for( auto it = A->begin(); it != A->end(); it++ ) {
    if( it->first.second != last_req ) {
      last_req = it->first.second;
      rowsNeeded[last_req/block_size].push_back(last_req);
    }
  }
  // sort these and flatten for the Alltoall
  int_e numNeeded[P];
  int_e reqOffsets[P];
  int_e totalNeeded = 0;
  for( int i = 0; i < P; i++ ) {
    numNeeded[i] = rowsNeeded[i].size();
    reqOffsets[i] = totalNeeded;
    totalNeeded += numNeeded[i];
    std::sort(rowsNeeded[i].begin(), rowsNeeded[i].end());
  }
  int_d *rowRequest = new int_d[totalNeeded];
  for( int i = 0; i < P; i++ )
    for( int j = 0; j < numNeeded[i]; j++ )
      rowRequest[reqOffsets[i]+j] = rowsNeeded[i][j];

  tfind += read_timer();

  MPI_Datatype MPI_ENTRY;
  MPI_Type_contiguous( sizeof(Entry), MPI_CHAR, &MPI_ENTRY );
  MPI_Type_commit( &MPI_ENTRY );

  MPI_Datatype MPI_INT_D;
  MPI_Type_contiguous( sizeof(int_d), MPI_CHAR, &MPI_INT_D );
  MPI_Type_commit( &MPI_INT_D );

  MPI_Datatype MPI_INT_E;
  MPI_Type_contiguous( sizeof(int_e), MPI_CHAR, &MPI_INT_E );
  MPI_Type_commit( &MPI_INT_E );

  double tcomm = -read_timer();

  // send and receive the requests of which rows of B are needed
  int_e requestedCounts[P];
  MPI_Alltoall( numNeeded, 1, MPI_INT_E, requestedCounts, 1, MPI_INT_E, MPI_COMM_WORLD );
  int_e requestedOffsets[P];
  requestedOffsets[0] = 0;
  for( int i = 1; i < P; i++ )
    requestedOffsets[i] = requestedOffsets[i-1]+requestedCounts[i-1];
  int_d *rowsRequested = new int_d[requestedOffsets[P-1]+requestedCounts[P-1]];
  MPI_Alltoallv( rowRequest, numNeeded, reqOffsets, MPI_INT_D, rowsRequested, requestedCounts, requestedOffsets, MPI_INT_D, MPI_COMM_WORLD );
  tcomm += read_timer();
  delete[] rowRequest;

  tfind -= read_timer();
  //gather the requested rows of B
  Matrix *entriesToSend = new Matrix[P];
  vector<pair<int,int_d> > next;
  int is[P];
  for( int i = 0; i < P; i++ )
    is[i] = 0;
  // initial population of the heap
  for( int i = 0; i < P; i++ )
    if( requestedCounts[i] > is[i] ) {
      next.push_back( make_pair( i, rowsRequested[requestedOffsets[i]+is[i]] ) );
      is[i]++;
    }
  make_heap(next.begin(), next.end(), ReverseRowMajorPair );

  auto it = B->begin();
  pop_heap( next.begin(), next.end(), ReverseRowMajorPair );
  while( !next.empty() && it != B->end() ) {
    // move forward in the matrix if the current row is too low
    while( it != B->end() && it->first.first < next.back().second ) {
      it++;
    }
    
    // move forward in the requests if the current row is too high
    while( it != B->end() && !next.empty() && it->first.first > next.back().second ) {
      int i = next.back().first;
      if( requestedCounts[i] > is[i] ) {
	next.back().second = rowsRequested[requestedOffsets[i]+is[i]];
	push_heap(next.begin(), next.end(), ReverseRowMajorPair );
	is[i]++;
	pop_heap(next.begin(), next.end(), ReverseRowMajorPair);
      } else {
	next.pop_back();
	if( !next.empty() )
	  pop_heap(next.begin(), next.end(), ReverseRowMajorPair );
      }
    }
    
    // add requested row
    if( !next.empty() && it != B->end() && it->first.first == next.back().second ) {
      int i = next.back().first;
      for( auto it2 = it; it2 != B->end() && it2->first.first == it->first.first; it2++ ) {
	entriesToSend[i].push_back( *it2 );
      }
      // advance in the requests
      if( requestedCounts[i] > is[i] ) {
	next.back().second = rowsRequested[requestedOffsets[i]+is[i]];
	push_heap(next.begin(), next.end(), ReverseRowMajorPair );
	is[i]++;
	pop_heap(next.begin(), next.end(), ReverseRowMajorPair);
      } else {
	next.pop_back();
	if( !next.empty() )
	  pop_heap(next.begin(), next.end(), ReverseRowMajorPair );
      }
    }
  }
  delete[] rowsRequested;

  // flatten the requested rows of B for sending
  int sendSizes[P];
  int sendOffsets[P];
  int totalSend = 0;
  for( int i = 0; i < P; i++ ) {
    sendOffsets[i] = totalSend;
    sendSizes[i] = entriesToSend[i].size();
    totalSend += sendSizes[i];
  }
  Entry *sendBuf = new Entry[totalSend];
  for( int i = 0; i < P; i++ )
    std::copy( entriesToSend[i].begin(), entriesToSend[i].end(), sendBuf+sendOffsets[i] );
  delete[] entriesToSend;

  tfind += read_timer();
  tcomm -= read_timer();

  int recvSizes[P];
  MPI_Alltoall( sendSizes, 1, MPI_INT, recvSizes, 1, MPI_INT, MPI_COMM_WORLD );
  int recvOffsets[P];
  int totalRecv = 0;
  for( int i = 0; i < P; i++ ) {
    recvOffsets[i] = totalRecv;
    totalRecv += recvSizes[i];
  }
  Entry *recvBuf = new Entry[totalRecv];
  MPI_Alltoallv( sendBuf, sendSizes, sendOffsets, MPI_ENTRY, 
		 recvBuf, recvSizes, recvOffsets, MPI_ENTRY, MPI_COMM_WORLD );

  // do the actual multiplication
  tcomm += read_timer();
  double tcomp = -read_timer();
  Matrix *C = local_multiply( A->data(), A->size(), recvBuf, totalRecv );
  tcomp += read_timer();
  Matrix *ret = sortDedup( C, CompColMajorEntry, true, times );

  if( times ) {
    times->push_back(tfind);
    times->push_back(tcomm);
    times->push_back(tcomp);
  }

  return ret;
}
