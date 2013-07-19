#include "communication.h"
#include "sizes.h"
#include "library.h"
#include <omp.h>

#define TAG_RED2 1
#define TAG_RED8 2
#define TAG_RED  4
#define TAG_DIST 3 
/*
  We assume that there are P=2^k processors.  At all times a power of 2 processors are working together on a task, and they are contiguously numbered.  That is, if rank i is working on a task with (x-1) other processors, those x processors have ranks floor(i/x)*x,...,(floor(i/x)+1)*x-1
 */

int rank;

int getRank() {
  return rank;
}

void initCommunication( int *argc, char ***argv ) {
  MPI_Init(argc, argv);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  omp_set_num_threads(NUM_THREADS);
}

int getRelativeRank( int xOld, int xNew ) {
  // the ranks involved in xOld are (rank/xOld)*xOld..(rank/xOld+1)*xOld-1
  return (rank - (rank/xOld)*xOld)/xNew;
}

// These could be made non-blocking, with followup functions if we want
// args have k entries, which are the k sources.  The ns are the sizes of the args; everything a given processor receieves is the same size, ns[gp]
void reduceBy( int k, int x, double **args, double *out, int* ns ) {
  int xh = x/k;
  int Proc0 = (rank/x)*x;
  int gp = (rank-Proc0)/xh;
  int offset = (rank-Proc0)%xh;
  
  double *buffer = (double*) malloc( ns[gp]*(k-1)*sizeof(double) );
  MPI_Request req[2*(k-1)];

  for( int i = 0; i < k; i++ ) {
    int target = Proc0+offset+xh*i;
    if( target < rank ) {
      MPI_Isend( args[i], ns[i], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+i );
      MPI_Irecv( buffer+i*ns[gp], ns[gp], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+k-1+i );
    } else if( target > rank ) {
      MPI_Isend( args[i], ns[i], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+i-1 );
      MPI_Irecv( buffer+(i-1)*ns[gp], ns[gp], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+k-2+i );
    }
  }

  MPI_Waitall( 2*(k-1), req, MPI_STATUSES_IGNORE );

  // Interleave the k pieces
  for( int i = 0; i < ns[gp]; i++ )
    for( int j = 0; j < k; j++ ) {
      if( j < gp )
	*(out++) = buffer[i+j*ns[gp]];
      else if( j == gp )
	*(out++) = args[j][i];
      else
	*(out++) = buffer[i+(j-1)*ns[gp]];	
    }
  free(buffer);
}

void iReduceBy1( int k, int x, double **args, double *out, int* ns, MPI_Request *&req, double *&buffer ) {
  int xh = x/k;
  int Proc0 = (rank/x)*x;
  int gp = (rank-Proc0)/xh;
  int offset = (rank-Proc0)%xh;
  
  buffer = (double*) malloc( ns[gp]*(k-1)*sizeof(double) );
  
  //MPI_Request req[2*(k-1)];
  req = new MPI_Request[2*(k-1)];

  for( int i = 0; i < k; i++ ) {
    int target = Proc0+offset+xh*i;
    if( target < rank ) {
      MPI_Isend( args[i], ns[i], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+i );
      MPI_Irecv( buffer+i*ns[gp], ns[gp], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+k-1+i );
    } else if( target > rank ) {
      MPI_Isend( args[i], ns[i], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+i-1 );
      MPI_Irecv( buffer+(i-1)*ns[gp], ns[gp], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+k-2+i );
    }
  }
}

void iReduceBy2( int k, int x, double **args, double *out, int* ns, MPI_Request *&req, double *&buffer ) {
  int xh = x/k;
  int Proc0 = (rank/x)*x;
  int gp = (rank-Proc0)/xh;
  MPI_Waitall( 2*(k-1), req, MPI_STATUSES_IGNORE );

  // Interleave the k pieces
  for( int i = 0; i < ns[gp]; i++ )
    for( int j = 0; j < k; j++ ) {
      if( j < gp )
	*(out++) = buffer[i+j*ns[gp]];
      else if( j == gp )
	*(out++) = args[j][i];
      else
	*(out++) = buffer[i+(j-1)*ns[gp]];	
    }
  free(buffer);
}

void expandBy( int k, int x, double **args, double *out, int* ns ) {
  int xh = x/k;
  int Proc0 = (rank/x)*x;
  int gp = (rank-Proc0)/xh;
  int offset = (rank-Proc0)%xh;

  double *buffer = (double*) malloc( (k-1)*ns[gp]*sizeof(double) );
  MPI_Request req[(k-1)*2];

  // Interleave the k pieces
  for( int i = 0; i < ns[gp]; i++ )
    for( int j = 0; j < k; j++ ) {
      if( j < gp )
	buffer[i+j*ns[gp]] = *(out++);
      else if( j == gp )
	args[j][i] = *(out++);
      else
	buffer[i+(j-1)*ns[gp]] = *(out++);	
    }


  for( int i = 0; i < k; i++ ) {
    int target = Proc0+offset+xh*i;
    if( target < rank ) {
      MPI_Irecv( args[i], ns[i], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+i );
      MPI_Isend( buffer+i*ns[gp], ns[gp], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+k-1+i );
    } else if( target > rank ) {
      MPI_Irecv( args[i], ns[i], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+i-1 );
      MPI_Isend( buffer+(i-1)*ns[gp], ns[gp], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+k-2+i );
    }
  }

  MPI_Waitall( 2*(k-1), req, MPI_STATUSES_IGNORE );
  free(buffer);

}

void iExpandBy1( int k, int x, double **args, double *out, int* ns, MPI_Request *&req, double *&buffer ) {
  int xh = x/k;
  int Proc0 = (rank/x)*x;
  int gp = (rank-Proc0)/xh;
  int offset = (rank-Proc0)%xh;

  buffer = (double*) malloc( (k-1)*ns[gp]*sizeof(double) );
  req = new MPI_Request[(k-1)*2];

  // Interleave the k pieces
  for( int i = 0; i < ns[gp]; i++ )
    for( int j = 0; j < k; j++ ) {
      if( j < gp )
	buffer[i+j*ns[gp]] = *(out++);
      else if( j == gp )
	args[j][i] = *(out++);
      else
	buffer[i+(j-1)*ns[gp]] = *(out++);	
    }


  for( int i = 0; i < k; i++ ) {
    int target = Proc0+offset+xh*i;
    if( target < rank ) {
      MPI_Irecv( args[i], ns[i], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+i );
      MPI_Isend( buffer+i*ns[gp], ns[gp], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+k-1+i );
    } else if( target > rank ) {
      MPI_Irecv( args[i], ns[i], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+i-1 );
      MPI_Isend( buffer+(i-1)*ns[gp], ns[gp], MPI_DOUBLE, target, TAG_RED, MPI_COMM_WORLD, req+k-2+i );
    }
  }
}

void iExpandBy2( int k, int x, double **args, double *out, int* ns, MPI_Request *&req, double *&buffer ) {

  MPI_Waitall( 2*(k-1), req, MPI_STATUSES_IGNORE );
  free(buffer);

}

void distFrom1ProcSq( double *A1, double *ADist, int n, int r, int P ) {
  if( r == 0 ) {
    int sqSize = getSizeSq( r, P );
    if( rank == 0 ) {
      for( int i = 0; i < sqSize*P; i++ ) {
	int target = i%P;
	if( target == 0 )
	  *(ADist++) = A1[i];
	else
	  MPI_Send( A1+i, 1, MPI_DOUBLE, target, TAG_DIST, MPI_COMM_WORLD );
      }
    } else {
      for( int i = 0; i < sqSize; i++ )
	MPI_Recv( ADist++, 1, MPI_DOUBLE, 0, TAG_DIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }
  } else {
    int nh = n/2;
    int sqOffset = getSizeSq( r-1, P );
    distFrom1ProcSq( A1, ADist, nh, r-1, P );
    distFrom1ProcSq( A1+nh*nh, ADist+sqOffset, nh, r-1, P );
    distFrom1ProcSq( A1+2*nh*nh, ADist+2*sqOffset, nh, r-1, P );
    distFrom1ProcSq( A1+3*nh*nh, ADist+3*sqOffset, nh, r-1, P );
  }
}

void distFrom1ProcTri( double *A1, double *ADist, int n, int r, int P ) {
  if( r == 0 ) {
    int triSize = getSizeTri( r, P );
    if( rank == 0 ) {
      for( int i = 0; i < triSize*P; i++ ) {
	int target = i%P;
	if( target == 0 )
	  *(ADist++) = A1[i];
	else
	  MPI_Send( A1+i, 1, MPI_DOUBLE, target, TAG_DIST, MPI_COMM_WORLD );
      }
    } else {
      for( int i = 0; i < triSize; i++ )
	MPI_Recv( ADist++, 1, MPI_DOUBLE, 0, TAG_DIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }
  } else {
    int nh = n/2;
    int sqOffset = getSizeSq( r-1, P );
    int triOffset = getSizeTri( r-1, P );
    distFrom1ProcTri( A1, ADist, nh, r-1, P );
    distFrom1ProcSq( A1+nh*(nh+1)/2, ADist+triOffset, nh, r-1, P );
    distFrom1ProcTri( A1+nh*(nh+1)/2+nh*nh, ADist+triOffset+sqOffset, nh, r-1, P );
  }
}

void colTo1ProcSq( double *A1, double *ADist, int n, int r, int P ) {
  if( r == 0 ) {
    int sqSize = getSizeSq( r, P );
    if( rank == 0 ) {
      for( int i = 0; i < sqSize*P; i++ ) {
	int target = i%P;
	if( target == 0 )
	  A1[i] = *(ADist++);
	else
	  MPI_Recv( A1+i, 1, MPI_DOUBLE, target, TAG_DIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      }
    } else {
      for( int i = 0; i < sqSize; i++ )
	MPI_Send( ADist++, 1, MPI_DOUBLE, 0, TAG_DIST, MPI_COMM_WORLD );
    }
  } else {
    int nh = n/2;
    int sqOffset = getSizeSq( r-1, P );
    colTo1ProcSq( A1, ADist, nh, r-1, P );
    colTo1ProcSq( A1+nh*nh, ADist+sqOffset, nh, r-1, P );
    colTo1ProcSq( A1+2*nh*nh, ADist+2*sqOffset, nh, r-1, P );
    colTo1ProcSq( A1+3*nh*nh, ADist+3*sqOffset, nh, r-1, P );
  }
}

void colTo1ProcTri( double *A1, double *ADist, int n, int r, int P ) {
  if( r == 0 ) {
    int triSize = getSizeTri( r, P );
    if( rank == 0 ) {
      for( int i = 0; i < triSize*P; i++ ) {
	int target = i%P;
	if( target == 0 )
	  A1[i] = *(ADist++);
	else
	  MPI_Recv( A1+i, 1, MPI_DOUBLE, target, TAG_DIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
      }
    } else {
      for( int i = 0; i < triSize; i++ )
	MPI_Send( ADist++, 1, MPI_DOUBLE, 0, TAG_DIST, MPI_COMM_WORLD );
    }
  } else {
    int nh = n/2;
    int sqOffset = getSizeSq( r-1, P );
    int triOffset = getSizeTri( r-1, P );
    colTo1ProcTri( A1, ADist, nh, r-1, P );
    colTo1ProcSq( A1+nh*(nh+1)/2, ADist+triOffset, nh, r-1, P );
    colTo1ProcTri( A1+nh*(nh+1)/2+nh*nh, ADist+triOffset+sqOffset, nh, r-1, P );
  }
}

