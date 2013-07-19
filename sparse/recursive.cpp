#include "recursive.h"
#include "local-multiply.h"
#include "library.h"
#include <mpi.h>
#include <assert.h>

int recDepth;
int *pattern;
MPI_Comm *commAtLevel;
int P;
MPI_Datatype MPI_ENTRY;
MPI_Datatype MPI_INT_E;

double comp = 0;
double comm_ab = 0;
double comm_c = 0;
double shuffle_ab = 0;
double shuffle_c = 0;

MPI_Comm *initCommunication( int r, int *patt ) {
  recDepth = r;
  pattern = new int[r];
  for( int i = 0; i < r; i++ )
    pattern[i] = patt[r-i-1];
  commAtLevel = new MPI_Comm[r];

  MPI_Type_contiguous( sizeof(int_e), MPI_CHAR, &MPI_INT_E );
  MPI_Type_commit( &MPI_INT_E );

  MPI_Type_contiguous( sizeof(Entry), MPI_CHAR, &MPI_ENTRY );
  MPI_Type_commit( &MPI_ENTRY );

  int rank;
  MPI_Group initialGroup;
  MPI_Comm_group( MPI_COMM_WORLD, &initialGroup );
  MPI_Group gp;
  MPI_Comm cm;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &P );
  assert( (1<<(2*r)) == P );
  for( int i = 0; i < r; i++ ) {
      int spread = 1<<(2*i);
      int color = rank/(4*spread)*spread + (rank%spread);
      MPI_Comm_split( MPI_COMM_WORLD, color, 0, commAtLevel+i );
    /*
    for( int j = 0; j < P/4; j++ ) {
      int ranks[4];
      int spread = 1<<(2*i);
      int first = (j/spread)*(spread*4)+j%spread;
      ranks[0] = first;
      ranks[1] = first+spread;
      ranks[2] = first+2*spread;
      ranks[3] = first+3*spread;
      MPI_Group_incl( initialGroup, 4, ranks, &gp );
      MPI_Comm_create( MPI_COMM_WORLD, gp, &cm );
      if( ranks[0] == rank || ranks[1] == rank || ranks[2] == rank || ranks[3] == rank ) {
	commAtLevel[i] = cm;
      }
      MPI_Group_free( &gp );
    }
    */
  }
  return commAtLevel;
}

Matrix *recursiveMultiply( Matrix *A, Matrix *B, int log2n, vector<double> *times ) {
  auto C = recMM( A->data(), A->size(), B->data(), B->size(), ((int_d)1)<<log2n, log2n, 0, 0, recDepth );
  Matrix *ret = sortDedup( C.first, C.second, CompColMajorEntry, false, times );
  if( times ) {
    times->push_back( comp );
    times->push_back( comm_ab );
    times->push_back( comm_c );
    times->push_back( shuffle_ab );
    times->push_back( shuffle_c );
  }
  return ret;
}

pair<Entry*,int_e> recMM( Entry *A, int_e sA, Entry *B, int_e sB, int_d m, int log2m, int_d mStartA, int_d mStartB, int recLevel ) {
  if( recLevel > 0 ) {
    if( pattern[recLevel-1] == 0 )
      return splitC( A, sA, B, sB, m, log2m, mStartA, mStartB, recLevel );
    return splitAB( A, sA, B, sB, m, log2m, mStartA, mStartB, recLevel );
  }
  comp -= read_timer();
  Matrix *C = local_multiply( A, sA, B, sB );
  comp += read_timer();
  // don't sort it, that will happen locally at the end
  // this is a memory leak
  return make_pair( C->data(), C->size() );
}

/*
  commAtLevel[recdepth-1] should be four processors that are as far apart as possible
  But they should own a contiguous piece of A and B
  So the processor owning block columns of A is:
  0 P/4 P/2 3P/4  P/8 3P/8 5P/8 7P/8  ....
 */

/*
  This does the split
  A1  B1 B2  C11 C12
  A2         C21 C22
  On input, A should be in block column layout, B in block row layout

  A and B are both of size m, but start at different places
 */

pair<Entry*,int_e> splitC( Entry *A, int_e sA, Entry *B, int_e sB, int_d m, int log2m, int_d mStartA, int_d mStartB, int recLevel ) {

  Entry *LA, *LB;
  int_e sLA, sLB;
  int_d mo2 = m/2;

  int_d mHalfA = mStartA+mo2;
  {  
    shuffle_c -= read_timer();
    Matrix A1, A2;
    A1.reserve(sA);
    for( int_e i = 0; i < sA; i++ ) {
      if( A[i].first.first >= mHalfA )
	A2.push_back( A[i] );
      else
	A1.push_back( A[i] );
    }
    int_e sendCounts[] = {int_e(A1.size()), int_e(A1.size()), int_e(A2.size()), int_e(A2.size())};
    int_e sendOffsets[] = {0, 0, int_e(A1.size()), int_e(A1.size())};
    A1.insert( A1.end(), A2.begin(), A2.end() );
    shuffle_c += read_timer();
    int_e recvCounts[4], recvOffsets[4];
    MPI_Comm comm = commAtLevel[recLevel-1];
    comm_c -= read_timer();
    MPI_Alltoall( sendCounts, 1, MPI_INT_E, recvCounts, 1, MPI_INT_E, comm );
    recvOffsets[0] = 0;
    recvOffsets[1] = recvCounts[0];
    recvOffsets[2] = recvCounts[1] + recvOffsets[1];
    recvOffsets[3] = recvCounts[2] + recvOffsets[2];
    sLA = recvOffsets[3]+recvCounts[3];
    LA = new Entry[sLA];
    MPI_Alltoallv( A1.data(), sendCounts, sendOffsets, MPI_ENTRY, LA, recvCounts, recvOffsets, MPI_ENTRY, comm );
    comm_c += read_timer();
  }

  int_d mHalfB = mStartB+mo2;
  {
    Matrix B1, B2;
    B1.reserve( sB );
    shuffle_c -= read_timer();
    for( int_e i = 0; i < sB; i++ ) {
      if( B[i].first.second >= mHalfB )
	B2.push_back( B[i] );
      else
	B1.push_back( B[i] );
    }
    int_e sendCounts[] = {int_e(B1.size()), int_e(B2.size()), int_e(B1.size()), int_e(B2.size())};
    int_e sendOffsets[] = {0, int_e(B1.size()), 0, int_e(B1.size())};
    B1.insert( B1.end(), B2.begin(), B2.end() );
    int_e recvCounts[4], recvOffsets[4];
    shuffle_c += read_timer();
    comm_c -= read_timer();
    MPI_Comm comm = commAtLevel[recLevel-1];
    MPI_Alltoall( sendCounts, 1, MPI_INT_E, recvCounts, 1, MPI_INT_E, comm );
    recvOffsets[0] = 0;
    recvOffsets[1] = recvCounts[0];
    recvOffsets[2] = recvCounts[1] + recvOffsets[1];
    recvOffsets[3] = recvCounts[2] + recvOffsets[2];
    sLB = recvOffsets[3]+recvCounts[3];
    LB = new Entry[sLB];
    MPI_Alltoallv( B1.data(), sendCounts, sendOffsets, MPI_ENTRY, LB, recvCounts, recvOffsets, MPI_ENTRY, comm );
    comm_c += read_timer();
  }

  int rrank;
  MPI_Comm comm = commAtLevel[recLevel-1];
  MPI_Comm_rank( comm, &rrank );
  int_d newStartA, newStartB;
  if( rrank == 0 || rrank == 1 )
    newStartA = mStartA;
  else
    newStartA = mHalfA;
  if( rrank == 0 || rrank == 2 )
    newStartB = mStartB;
  else
    newStartB = mHalfB;

  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  pair<Entry*,int_e> C = recMM( LA, sLA, LB, sLB, mo2, log2m-1, newStartA, newStartB, recLevel-1 );
  delete[] LA;
  delete[] LB;
  return C;
}

/*
  This is for the split
                B1
  A1 A2 A3 A4   B2  C
                B3
		B4
  At this point, C is m x m
  A is block column and B is block row
  Actually, the layout may be more complicated, but columns of A match to rows of B
 */

pair<Entry*,int_e> splitAB( Entry *A, int_e sA, Entry *B, int_e sB, int_d m, int log2m, int_d mStartA, int_d mStartB, int recLevel ) {
  pair<Entry*,int_e> lCpair = recMM( A, sA, B, sB, m, log2m, mStartA, mStartB, recLevel-1 );
  Entry *lC = lCpair.first;
  int_e slC = lCpair.second;

  int_e recvSize;
  Entry *buf;
  {
    // split C into which part goes to which processor
    shuffle_ab -= read_timer();    
    Matrix Cs[4];
    Cs[0].reserve( slC );
    for( int_e i = 0; i < slC; i++ ) {
      int_d col = lC[i].first.second;
      int_d row = lC[i].first.first;
      bool targetR = !!(row & (((int_d)1) << (log2m-recLevel)));
      bool targetC = !!(col & (((int_d)1) << (log2m-recLevel)));
      Cs[targetC+2*targetR].push_back(lC[i]);
    }
    int_e sendOffsets[] = {0, int_e(Cs[0].size()), int_e(Cs[0].size()+Cs[1].size()), int_e(Cs[0].size()+Cs[1].size()+Cs[2].size())};
    int_e sendCounts[] = {int_e(Cs[0].size()), int_e(Cs[1].size()), int_e(Cs[2].size()), int_e(Cs[3].size())};
    Cs[0].insert( Cs[0].end(), Cs[1].begin(), Cs[1].end() );
    Cs[0].insert( Cs[0].end(), Cs[2].begin(), Cs[2].end() );
    Cs[0].insert( Cs[0].end(), Cs[3].begin(), Cs[3].end() );
    int_e recvCounts[4], recvOffsets[4];
    shuffle_ab += read_timer();
    comm_ab -= read_timer();
    MPI_Comm comm = commAtLevel[recLevel-1];
    MPI_Alltoall( sendCounts, 1, MPI_INT_E, recvCounts, 1, MPI_INT_E, comm );
    recvOffsets[0] = 0;
    recvOffsets[1] = recvCounts[0];
    recvOffsets[2] = recvCounts[1] + recvOffsets[1];
    recvOffsets[3] = recvCounts[2] + recvOffsets[2];
    recvSize = recvOffsets[3]+recvCounts[3];
    buf = new Entry[recvSize];
    MPI_Alltoallv( Cs[0].data(), sendCounts, sendOffsets, MPI_ENTRY, buf, recvCounts, recvOffsets, MPI_ENTRY, comm );
    comm_ab += read_timer();
  }
  return make_pair( buf, recvSize );
}

