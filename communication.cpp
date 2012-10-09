#include "communication.h"
#include "tags.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

MPI_Comm *commAtLevel, baseComm;
int log7nProcs, rank, pFactor, base;

void initCommunication(int *argc, char ***argv, int randomize ) {
  MPI_Init(argc, argv);
  int nProcs;
  MPI_Comm_size( MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);

  // randomize if requested
  int rankRedirect[nProcs];
  int rankRedirectInverse[nProcs];

  if( randomize ) {
    // generate the random arrangement
    srand48(randomize);
    int taken[nProcs];
    for( int i = 0; i < nProcs; i++ )
      taken[i] = 0;
    for( int i = 0; i < nProcs; i++ ) {
      double rr = drand48();
      int r = (int) (rr*(nProcs-i));
      for( int j = 0; j <= r; j++ )
	if( taken[j] )
	  r++;
      taken[r] = 1;
      rankRedirect[i] = r;
      rankRedirectInverse[r] = i;
    }
  } else {
    for( int i = 0; i < nProcs; i++ ) {
      rankRedirect[i] = i;
      rankRedirectInverse[i] = i;
    }
  }

  // check that nProcs is a power of 7
  int i = nProcs;
  log7nProcs = 0;
  while( i > 1 ) {
    if( i % SEVEN != 0 ) {
      break;
    }
    i = i/SEVEN;
    log7nProcs++;
  }
  pFactor = i;
  if( rank == 0 )
    printf( "Running with %d=%d*%d^%d processes\n", nProcs,pFactor,SEVEN,log7nProcs );

  base = nProcs / pFactor;
  MPI_Group initialGroup;
  MPI_Comm_group( MPI_COMM_WORLD, &initialGroup );
  // set up the communicator for the traditional matrix multiply at the bottom
  if( pFactor != 1 ) {
    int ranks[pFactor];
    for( int i = 0, proc = rankRedirectInverse[rank] % base; i < pFactor; i++, proc+=base ) {
      ranks[i] = rankRedirect[proc];
    }
    MPI_Group gp;
    MPI_Group_incl( initialGroup, pFactor, ranks, &gp );
    MPI_Comm_create( MPI_COMM_WORLD, gp, &baseComm );
  }

  if( log7nProcs == 0 ) {
    // single processor (wrt strassen)
    return;
  }
  MPI_Group groupAtLevel[log7nProcs];
  for( int i = 0, procsThisLevel = base; i < log7nProcs; i++, procsThisLevel /= SEVEN ) {
    int firstProc = (rankRedirectInverse[rank]/procsThisLevel)*procsThisLevel;
    int crossFirstProc = rankRedirectInverse[rank];
    crossFirstProc = (crossFirstProc-firstProc)%(procsThisLevel/SEVEN)+firstProc;
    int ranks[SEVEN];
    for( int j = 0; j < SEVEN; j++ ) {
      ranks[j] = rankRedirect[crossFirstProc+j*(procsThisLevel/SEVEN)];
    }
    MPI_Group_incl( initialGroup, SEVEN, ranks, groupAtLevel+i );
  }
  commAtLevel = new MPI_Comm[log7nProcs];
  for( int i = 0; i < log7nProcs; i++ ) {
    MPI_Comm_create( MPI_COMM_WORLD, groupAtLevel[i], commAtLevel+i );
  }

}

void initCommunicationAlt(int *argc, char ***argv, int randomize ) {
  MPI_Init(argc, argv);
  int nProcs;
  MPI_Comm_size( MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);

  // randomize if requested
  int rankRedirect[nProcs];
  int rankRedirectInverse[nProcs];

  if( randomize ) {
    // generate the random arrangement
    srand48(randomize);
    int taken[nProcs];
    for( int i = 0; i < nProcs; i++ )
      taken[i] = 0;
    for( int i = 0; i < nProcs; i++ ) {
      double rr = drand48();
      int r = (int) (rr*(nProcs-i));
      for( int j = 0; j <= r; j++ )
	if( taken[j] )
	  r++;
      taken[r] = 1;
      rankRedirect[i] = r;
      rankRedirectInverse[r] = i;
    }
  } else {
    for( int i = 0; i < nProcs; i++ ) {
      rankRedirect[i] = i;
      rankRedirectInverse[i] = i;
    }
  }

  // check that nProcs is a power of 7
  int i = nProcs;
  log7nProcs = 0;
  while( i > 1 ) {
    if( i % SEVEN != 0 ) {
      break;
    }
    i = i/SEVEN;
    log7nProcs++;
  }
  pFactor = i;
  if( rank == 0 )
    printf( "Running with %d=%d*%d^%d processes\n", nProcs,pFactor,SEVEN,log7nProcs );

  base = nProcs / pFactor;
  MPI_Group initialGroup;
  MPI_Comm_group( MPI_COMM_WORLD, &initialGroup );
  // set up the communicator for the traditional matrix multiply at the bottom
  if( pFactor != 1 ) {
    int ranks[pFactor];
    for( int i = 0, proc = rankRedirectInverse[(rank/pFactor)*pFactor]; i < pFactor; i++, proc++ ) {
      ranks[i] = rankRedirect[proc];
    }
    MPI_Group gp;
    MPI_Group_incl( initialGroup, pFactor, ranks, &gp );
    MPI_Comm_create( MPI_COMM_WORLD, gp, &baseComm );
  }

  if( log7nProcs == 0 ) {
    // single processor (wrt strassen)
    return;
  }
  MPI_Group groupAtLevel[log7nProcs];
  for( int i = 0, procsThisLevel = base*pFactor; i < log7nProcs; i++, procsThisLevel /= SEVEN ) {
    int firstProc = (rankRedirectInverse[rank]/procsThisLevel)*procsThisLevel;
    int crossFirstProc = rankRedirectInverse[rank];
    crossFirstProc = (crossFirstProc-firstProc)%(procsThisLevel/SEVEN)+firstProc;
    int ranks[SEVEN];
    for( int j = 0; j < SEVEN; j++ ) {
      ranks[j] = rankRedirect[crossFirstProc+(j*(procsThisLevel/SEVEN))];
    }
    MPI_Group_incl( initialGroup, SEVEN, ranks, groupAtLevel+i );
  }
  commAtLevel = new MPI_Comm[log7nProcs];
  for( int i = 0; i < log7nProcs; i++ ) {
    MPI_Comm_create( MPI_COMM_WORLD, groupAtLevel[i], commAtLevel+i );
  }

}

int getRank() {
  return rank;
}

int getLog7nProcs() {
  return log7nProcs;
}

int getPFactor() {
  return pFactor;
}

MPI_Comm getComm() {
  return MPI_COMM_WORLD;
}

MPI_Comm getSummaComm( MatDescriptor desc ) {
  return baseComm;
}

// get the communicators used for gather and scatter when collapsing/expanding a column or a row
MPI_Comm getColComm( MatDescriptor desc1, MatDescriptor desc2 ) {
  int i = 0;
  while( desc1.nprocr > 1 )
    i++, desc1.nprocr/=SEVEN;
  return commAtLevel[log7nProcs-i];
}
MPI_Comm getRowComm( MatDescriptor desc1, MatDescriptor desc2 ) {
  int i = 0;
  while( desc1.nprocr > 1 )
    i++, desc1.nprocr/=SEVEN;
  return commAtLevel[(log7nProcs+1)/2-i];
}

// these next three divide gatherMatrices up into three parts.  First, non-blocking sends of three components; Second, sends of the other 4; Third the local re-arrangement.  The point is that the three components that require no additions to compute can be sent while the additions are being done

void scatter1( int target, int rank, double *S0, MatDescriptor sDesc, MPI_Comm comm, MPI_Request *reqs, int nproc, double *Work ) {  
  if( target < rank )
    MPI_Isend( S0, numEntriesPerProc(sDesc), MPI_DOUBLE, target, TAG_S0, comm, reqs+target );
  else if( target > rank ) 
    MPI_Isend( S0, numEntriesPerProc(sDesc), MPI_DOUBLE, target, TAG_S0, comm, reqs+target-1 );
  else {
    for( int source = 0; source < rank; source++ )
      MPI_Irecv( Work+source*numEntriesPerProc(sDesc), numEntriesPerProc(sDesc), MPI_DOUBLE, source, TAG_S0, comm, reqs+nproc-1+source );
    memcpy( Work+rank*numEntriesPerProc(sDesc), S0, numEntriesPerProc(sDesc)*sizeof(double) );
    for( int source = rank+1; source < nproc; source++ )
      MPI_Irecv( Work+source*numEntriesPerProc(sDesc), numEntriesPerProc(sDesc), MPI_DOUBLE, source, TAG_S0, comm, reqs+nproc-2+source );
  }  
}

void gather1( int source, int rank, double *S0, MatDescriptor sDesc, MPI_Comm comm, MPI_Request *reqs, int nproc, double *Work ) {  
  if( source < rank )
    MPI_Irecv( S0, numEntriesPerProc(sDesc), MPI_DOUBLE, source, TAG_S0, comm, reqs+source );
  else if( source > rank ) 
    MPI_Irecv( S0, numEntriesPerProc(sDesc), MPI_DOUBLE, source, TAG_S0, comm, reqs+source-1 );
  else {
    for( int source = 0; source < rank; source++ )
      MPI_Isend( Work+source*numEntriesPerProc(sDesc), numEntriesPerProc(sDesc), MPI_DOUBLE, source, TAG_S0, comm, reqs+nproc-1+source );
    memcpy( S0, Work+rank*numEntriesPerProc(sDesc), numEntriesPerProc(sDesc)*sizeof(double) );
    for( int source = rank+1; source < nproc; source++ )
      MPI_Isend( Work+source*numEntriesPerProc(sDesc), numEntriesPerProc(sDesc), MPI_DOUBLE, source, TAG_S0, comm, reqs+nproc-2+source );
  }  
}

// SPos[0], SPos[1], SPos[2] are integers between 0 and 7 that specify the target processors of S0, S1, S2 respectively.
void gatherMatrices1( MatDescriptor sDesc, double *S0, double *S1, double *S2, int *SPos, MatDescriptor tDesc, double *Work, MPI_Request **reqs ){
  // check the descriptors are sensible, and determine if we are collapsing columns or rows
#ifdef SANITY_CHECKS
  assert( sDesc.lda == tDesc.lda );
  assert( sDesc.bs == tDesc.bs );
  assert( sDesc.nproc == tDesc.nproc * SEVEN );
#endif
#ifdef SANITY_CHECKS
  bool collapseCols;
#endif
  MPI_Comm comm;
  if( sDesc.nprocr == tDesc.nprocr ) {
#ifdef SANITY_CHECKS
    collapseCols = true;
#endif
    comm = getColComm( sDesc, tDesc );
  } else {
#ifdef SANITY_CHECKS
    collapseCols = false;
#endif
    comm = getRowComm( sDesc, tDesc );
  }
  // it is assumed that when square, we collaps columns, and when not square, we collapse rows
#ifdef SANITY_CHECKS
  assert( (collapseCols && (sDesc.nprocr == sDesc.nprocc)) || (!collapseCols && (sDesc.nprocr != sDesc.nprocc)) );
#endif
  startTimer(TIMER_COMM);
  int rank;
  MPI_Comm_rank( comm, &rank );
  *reqs = new MPI_Request[2*(SEVEN-1)]; // we won't use all of these this function, but later
  scatter1( SPos[0], rank, S0, sDesc, comm, *reqs, SEVEN, Work );
  scatter1( SPos[1], rank, S1, sDesc, comm, *reqs, SEVEN, Work );
  scatter1( SPos[2], rank, S2, sDesc, comm, *reqs, SEVEN, Work );
  
  stopTimer(TIMER_COMM);
}

// now 4 more.  reqs better already be initialized by gatherMatrices1
void gatherMatrices2( MatDescriptor sDesc, double *S0, double *S1, double *S2, double *S3, int *SPos, MatDescriptor tDesc, double *Work, MPI_Request *reqs ){
  // check the descriptors are sensible, and determine if we are collapsing columns or rows
#ifdef SANITY_CHECKS
  assert( sDesc.lda == tDesc.lda );
  assert( sDesc.bs == tDesc.bs );
  assert( sDesc.nproc == tDesc.nproc * SEVEN );
#endif
#ifdef SANITY_CHECKS
  bool collapseCols;
#endif
  MPI_Comm comm;
  if( sDesc.nprocr == tDesc.nprocr ) {
#ifdef SANITY_CHECKS
    collapseCols = true;
#endif
    comm = getColComm( sDesc, tDesc );
  } else {
#ifdef SANITY_CHECKS
    collapseCols = false;
#endif
    comm = getRowComm( sDesc, tDesc );
  }
  // it is assumed that when square, we collaps columns, and when not square, we collapse rows
#ifdef SANITY_CHECKS
  assert( (collapseCols && (sDesc.nprocr == sDesc.nprocc)) || (!collapseCols && (sDesc.nprocr != sDesc.nprocc)) );
#endif
  startTimer(TIMER_COMM);
  int rank;
  MPI_Comm_rank( comm, &rank );
  scatter1( SPos[0], rank, S0, sDesc, comm, reqs, SEVEN, Work );
  scatter1( SPos[1], rank, S1, sDesc, comm, reqs, SEVEN, Work );
  scatter1( SPos[2], rank, S2, sDesc, comm, reqs, SEVEN, Work );
  scatter1( SPos[3], rank, S3, sDesc, comm, reqs, SEVEN, Work );

  stopTimer(TIMER_COMM);
}

// wait for the communication to finish, then do the local re-arrangement
void gatherMatrices3( MatDescriptor sDesc, MatDescriptor tDesc, double *Work, MPI_Request *reqs, double *T ){
  // check the descriptors are sensible, and determine if we are collapsing columns or rows
#ifdef SANITY_CHECKS
  assert( sDesc.lda == tDesc.lda );
  assert( sDesc.bs == tDesc.bs );
  assert( sDesc.nproc == tDesc.nproc * SEVEN );
#endif
  bool collapseCols;
  if( sDesc.nprocr == tDesc.nprocr ) {
    collapseCols = true;
  } else {
    collapseCols = false;
  }
#ifdef SANITY_CHECKS
  // it is assumed that when square, we collaps columns, and when not square, we collapse rows
  assert( (collapseCols && (sDesc.nprocr == sDesc.nprocc)) || (!collapseCols && (sDesc.nprocr != sDesc.nprocc)) );
#endif
  startTimer(TIMER_COMM);
  MPI_Waitall( 2*(SEVEN-1), reqs, MPI_STATUSES_IGNORE );
  delete [] reqs;
  increaseMessages((SEVEN-1)*2);
  increaseWords((SEVEN-1)*2*numEntriesPerProc(sDesc)*sizeof(double));
  stopTimer(TIMER_COMM);
  int bs2 = sDesc.bs*sDesc.bs;
  int entriesPerProc = numEntriesPerProc( sDesc );
  int copySize;
  if( collapseCols ) // if we are collapsing a column, interleave at every block
    copySize = bs2;
  else // if rows, interleave at the number of entries stored in a column of blocks
    copySize = sDesc.bs*sDesc.lda / ((1<<sDesc.nrec) * sDesc.nprocc);
  int copysPerProc = entriesPerProc / copySize;
  startTimer(TIMER_REARRANGE);
#pragma omp parallel for schedule(static, (copysPerProc+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < copysPerProc; i++ ) {
    for( int p = 0; p < SEVEN; p++ ) {
      memcpy( T + p*copySize + SEVEN*copySize*i, Work + p*entriesPerProc + copySize*i, copySize*sizeof(double) );
    }
  }
  stopTimer(TIMER_REARRANGE);
}

void gatherMatricesS0( MatDescriptor sDesc, MatDescriptor tDesc, MPI_Request **reqs ){
  // check the descriptors are sensible, and determine if we are collapsing columns or rows
#ifdef SANITY_CHECKS
  assert( sDesc.lda == tDesc.lda );
  assert( sDesc.bs == tDesc.bs );
  assert( sDesc.nproc_summa != 1 );
  assert( tDesc.nproc_summa == 1 );
  assert( sDesc.nproc == 1 );
  assert( tDesc.nproc == 1 );
#endif
  increaseMessages((sDesc.nproc_summa-1)*2);
  increaseWords((sDesc.nproc_summa-1)*2*numEntriesPerProc(sDesc)*sizeof(double));
  *reqs = new MPI_Request[2*(sDesc.nproc_summa-1)];
}

//reqs better already be initialized by gatherMatricesS0
void gatherMatricesS1( MatDescriptor sDesc, double **Ss, int *SPos, int sNum, MatDescriptor tDesc, double *Work, MPI_Request *reqs ){
  // check the descriptors are sensible, and determine if we are collapsing columns or rows
#ifdef SANITY_CHECKS
  assert( sDesc.lda == tDesc.lda );
  assert( sDesc.bs == tDesc.bs );
  assert( sDesc.nproc_summa != 1 );
  assert( tDesc.nproc_summa == 1 );
  assert( sDesc.nproc == 1 );
  assert( tDesc.nproc == 1 );
#endif
  MPI_Comm comm = getSummaComm( sDesc );
  startTimer(TIMER_COMM);
  int rank;
  MPI_Comm_rank( comm, &rank );
  for( int i = 0; i < sNum; i++ )
    scatter1( SPos[i], rank, Ss[i], sDesc, comm, reqs, sDesc.nproc_summa, Work );
  
  stopTimer(TIMER_COMM);
}

// wait for the communication to finish, do the re-arrangement
void gatherMatricesS2( MPI_Request *reqs, MatDescriptor sDesc, MatDescriptor tDesc, double *Work, double *T ){
  startTimer(TIMER_COMM);
  MPI_Waitall( 2*(sDesc.nproc_summa-1), reqs, MPI_STATUSES_IGNORE );
  delete [] reqs;
  stopTimer(TIMER_COMM);
  startTimer(TIMER_REARRANGE);
  int nblocks = powl(4, sDesc.nrec);
  int entriesPerProc = numEntriesPerProc(sDesc);
  int blocksize = numEntriesPerProc(sDesc)/nblocks;
  for( int b = 0; b < nblocks; b++ )
    for( int p = 0; p < sDesc.nproc_summa; p++ )
      memcpy( T + p*blocksize + sDesc.nproc_summa*blocksize*b, Work + p*entriesPerProc + blocksize*b, blocksize*sizeof(double) );
  stopTimer(TIMER_REARRANGE);
}

void scatterMatricesS0( MatDescriptor sDesc, MatDescriptor tDesc, MPI_Request **reqs, double *T, double *work ){
  // check the descriptors are sensible, and determine if we are collapsing columns or rows
#ifdef SANITY_CHECKS
  assert( sDesc.lda == tDesc.lda );
  assert( sDesc.bs == tDesc.bs );
  assert( sDesc.nproc_summa != 1 );
  assert( tDesc.nproc_summa == 1 );
  assert( sDesc.nproc == 1 );
  assert( tDesc.nproc == 1 );
#endif
  startTimer(TIMER_REARRANGE);
  int nblocks = powl(4, sDesc.nrec);
  int entriesPerProc = numEntriesPerProc(sDesc);
  int blocksize = numEntriesPerProc(sDesc)/nblocks;
  for( int b = 0; b < nblocks; b++ )
    for( int p = 0; p < sDesc.nproc_summa; p++ )
      memcpy( work + p*entriesPerProc + blocksize*b, T + p*blocksize + sDesc.nproc_summa*blocksize*b, blocksize*sizeof(double) );
  stopTimer(TIMER_REARRANGE);

  increaseMessages((sDesc.nproc_summa-1)*2);
  increaseWords((sDesc.nproc_summa-1)*2*numEntriesPerProc(sDesc)*sizeof(double));
  *reqs = new MPI_Request[2*(sDesc.nproc_summa-1)];
}

void scatterMatricesS1( MatDescriptor sDesc, double **Ss, int *SPos, int sNum, MatDescriptor tDesc, double *work, MPI_Request *reqs ){
  // check the descriptors are sensible, and determine if we are collapsing columns or rows
#ifdef SANITY_CHECKS
  assert( sDesc.lda == tDesc.lda );
  assert( sDesc.bs == tDesc.bs );
  assert( sDesc.nproc_summa != 1 );
  assert( tDesc.nproc_summa == 1 );
  assert( sDesc.nproc == 1 );
  assert( tDesc.nproc == 1 );
#endif

  MPI_Comm comm = getSummaComm( sDesc );

  startTimer(TIMER_COMM);
  int rank;
  MPI_Comm_rank( comm, &rank );

  for( int i = 0; i < sNum; i++ )
    gather1( SPos[i], rank, Ss[i], sDesc, comm, reqs, sDesc.nproc_summa, work );
  
  stopTimer(TIMER_COMM);
}

void scatterMatricesS2( MatDescriptor sDesc, MPI_Request *reqs ) {
  startTimer(TIMER_COMM);
  MPI_Waitall( 2*(sDesc.nproc_summa-1), reqs, MPI_STATUSES_IGNORE );
  delete [] reqs;
  stopTimer(TIMER_COMM);
}

void gatherMatrices( MatDescriptor sDesc, double *S1234567, MatDescriptor tDesc, double *T, double *Work ){
  // check the descriptors are sensible, and determine if we are collapsing columns or rows
#ifdef SANITY_CHECKS
  assert( sDesc.lda == tDesc.lda );
  assert( sDesc.bs == tDesc.bs );
  assert( sDesc.nproc == tDesc.nproc * SEVEN );
#endif
  bool collapseCols;
  if( sDesc.nprocr == tDesc.nprocr )
    collapseCols = true;
  else
    collapseCols = false;
  // it is assumed that when square, we collaps columns, and when not square, we collapse rows
#ifdef SANITY_CHECKS
  assert( (collapseCols && (sDesc.nprocr == sDesc.nprocc)) || (!collapseCols && (sDesc.nprocr != sDesc.nprocc)) );
#endif
  MPI_Comm comm;
  if( collapseCols )
    comm = getColComm( sDesc, tDesc );
  else
    comm = getRowComm( sDesc, tDesc );
  startTimer(TIMER_COMM);
  MPI_Alltoall( S1234567, numEntriesPerProc(sDesc), MPI_DOUBLE, Work, numEntriesPerProc(sDesc), MPI_DOUBLE, comm );
  stopTimer(TIMER_COMM);
  increaseMessages((SEVEN-1)*2);
  increaseWords((SEVEN-1)*2*numEntriesPerProc(sDesc)*sizeof(double));

  int bs2 = sDesc.bs*sDesc.bs;
  int entriesPerProc = numEntriesPerProc( sDesc );
  int copySize;
  if( collapseCols ) // if we are collapsing a column, interleave at every block
    copySize = bs2;
  else // if rows, interleave at the number of entries stored in a column of blocks
    copySize = sDesc.bs*sDesc.lda / ((1<<sDesc.nrec) * sDesc.nprocc);
  int copysPerProc = entriesPerProc / copySize;
  startTimer(TIMER_REARRANGE);
#pragma omp parallel for schedule(static, (copysPerProc+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < copysPerProc; i++ ) {
    for( int p = 0; p < SEVEN; p++ ) {
      memcpy( T + p*copySize + SEVEN*copySize*i, Work + p*entriesPerProc + copySize*i, copySize*sizeof(double) );
    }
  }
  stopTimer(TIMER_REARRANGE);
}

// this is just written as the opposite of gatherMatrices.
void scatterMatrices( MatDescriptor sDesc, double *S, MatDescriptor tDesc, double *T1234567, double *Work ){
  // check the descriptors are sensible, and determine if we are collapsing columns or rows
#ifdef SANITY_CHECKS
  assert( sDesc.lda == tDesc.lda );
  assert( sDesc.bs == tDesc.bs );
  assert( tDesc.nproc == sDesc.nproc * SEVEN );
#endif
  bool expandCols;
  if( sDesc.nprocr == tDesc.nprocr )
    expandCols = true;
  else
    expandCols = false;
  // it is assumed that when square, we expand rows, and when not square, we expand columns
#ifdef SANITY_CHECKS
  assert( (expandCols && (sDesc.nprocr != sDesc.nprocc)) || (!expandCols && (sDesc.nprocr == sDesc.nprocc)) );
#endif

  int bs2 = sDesc.bs*sDesc.bs;
  int entriesPerProc = numEntriesPerProc( tDesc );
  int copySize;
  if( expandCols ) 
    //copySize = tDesc.bs*tDesc.lda / ((1<<tDesc.nrec) * tDesc.nprocr);
    copySize = bs2;
  else
    copySize = tDesc.bs*tDesc.lda / ((1<<tDesc.nrec) * tDesc.nprocc);
  int copysPerProc = entriesPerProc / copySize;
  startTimer(TIMER_REARRANGE);
#pragma omp parallel for schedule(static, (copysPerProc+NUM_THREADS-1)/NUM_THREADS)
  for( int i = 0; i < copysPerProc; i++ ) {
      for( int p = 0; p < SEVEN; p++ ) {
	memcpy( Work + p*entriesPerProc + copySize*i, S + p*copySize + SEVEN*copySize*i, copySize*sizeof(double) );
      }
  }
  stopTimer(TIMER_REARRANGE);

  MPI_Comm comm;
  if( expandCols )
    comm = getColComm( tDesc, sDesc );
  else
    comm = getRowComm( tDesc, sDesc );

  startTimer(TIMER_COMM);
  MPI_Alltoall( Work, numEntriesPerProc(tDesc), MPI_DOUBLE, T1234567, numEntriesPerProc(tDesc), MPI_DOUBLE, comm );
  stopTimer(TIMER_COMM);
  increaseMessages((SEVEN-1)*2);
  increaseWords((SEVEN-1)*2*numEntriesPerProc(tDesc)*sizeof(double));

}

void sendBlock( MPI_Comm comm, int rank, int target, double *O, int bs, int source, double *I, int ldi ) {
  if( source == target ) {
    if( rank == source ) {
      for( int c = 0; c < bs; c++ ) {
	memcpy( O, I, bs*sizeof(double) );
	O += bs;
	I += ldi;
      }
    }
  } else {
    if( rank == source ) { // send
      for( int c = 0; c < bs; c++ ) {
	MPI_Send( I, bs, MPI_DOUBLE, target, TAG_SEND_BLOCK, comm );
	I += ldi;
      }
    }
    else if( rank == target ) { //receive
      for( int c = 0; c < bs; c++ ) {
	MPI_Status stat;
	MPI_Recv( O, bs, MPI_DOUBLE, source, TAG_SEND_BLOCK, comm, &stat );
	O += bs;
      }
    }
  }
}

void receiveBlock( MPI_Comm comm, int rank, int target, double *O, int bs, int source, double *I, int ldo ) {
  if( source == target ) {
    if( rank == source ) {
      for( int c = 0; c < bs; c++ ) {
	memcpy( O, I, bs*sizeof(double) );
	O += ldo;
	I += bs;
      }
    }
  } else {
    if( rank == source ) { // send
      for( int c = 0; c < bs; c++ ) {
	MPI_Send( I, bs, MPI_DOUBLE, target, TAG_RECEIVE_BLOCK, comm );
	I += bs;
      }
    }
    else if( rank == target ) { //receive
      for( int c = 0; c < bs; c++ ) {
	MPI_Status stat;
	MPI_Recv( O, bs, MPI_DOUBLE, source, TAG_RECEIVE_BLOCK, comm, &stat );
	O += ldo;
      }
    }
  }
}

void distributeFrom1ProcRec( MatDescriptor desc, double *O, double *I, int ldi ) {
  if( desc.nrec == 0 ) { // base case; put the matrix block-cyclic layout
    MPI_Comm comm = getComm();
    int rank = getRank();
    int bs = desc.bs;
    int numBlocks = desc.lda / bs;
    assert( numBlocks % desc.nprocr == 0 );
    assert( numBlocks % desc.nprocc == 0 );
    assert( (numBlocks / desc.nprocr) % desc.nproc_summa == 0 );
    int nBlocksPerProcRow = numBlocks / desc.nprocr / desc.nproc_summa;
    int nBlocksPerProcCol = numBlocks / desc.nprocc;
    int nBlocksPerBase = numBlocks / desc.nproc_summa;
    for( int sp = 0; sp < desc.nproc_summa; sp++ ) {
      for( int i = 0; i < nBlocksPerProcRow; i++ ) {
	for( int rproc = 0; rproc < desc.nprocr; rproc++ ) {
	  for( int j = 0; j < nBlocksPerProcCol; j++ ) {
	    for( int cproc = 0; cproc < desc.nprocc; cproc++ ) {
	      int source = 0;
	      int target = cproc + rproc*desc.nprocc + sp*base;
	      // row and column of the beginning of the block in I
	      int row = j*(desc.nprocc*bs) + cproc*bs;
	      int col = i*(desc.nprocr*bs) + rproc*bs + sp*nBlocksPerBase*bs;
	      int offsetSource = row + col*ldi;
	      int offsetTarget = (j + i*nBlocksPerProcCol)*bs*bs;
	      sendBlock( comm, rank, target, O+offsetTarget, bs, source, I+offsetSource, ldi );
	    }
	  }
	}
      }
    }
  } else { // recursively call on each of four submatrices
    desc.nrec -= 1;
    desc.lda /= 2;
    int entriesPerQuarter = numEntriesPerProc(desc);
    // top left
    distributeFrom1ProcRec( desc, O, I, ldi );
    // bottom left
    distributeFrom1ProcRec( desc, O + entriesPerQuarter, I+desc.lda, ldi );
    // top right
    distributeFrom1ProcRec( desc, O + 2*entriesPerQuarter, I+desc.lda*ldi, ldi );
    // bottom right
    distributeFrom1ProcRec( desc, O + 3*entriesPerQuarter, I+desc.lda*ldi+desc.lda, ldi );
  }
}

void distributeFrom1Proc( MatDescriptor desc, double *O, double *I ) {
  distributeFrom1ProcRec( desc, O, I, desc.lda );
}

void collectTo1ProcRec( MatDescriptor desc, double *O, double *I, int ldo ) {
  if( desc.nrec == 0 ) { // base case; put the matrix block-cyclic layout
    MPI_Comm comm = getComm();
    int rank = getRank();
    int bs = desc.bs;
    int numBlocks = desc.lda / bs;
    assert( numBlocks % desc.nprocr == 0 );
    assert( numBlocks % desc.nprocc == 0 );
    assert( (numBlocks / desc.nprocr) % desc.nproc_summa == 0 );
    int nBlocksPerProcRow = numBlocks / desc.nprocr / desc.nproc_summa;
    int nBlocksPerProcCol = numBlocks / desc.nprocc;
    int nBlocksPerBase = numBlocks / desc.nproc_summa;
    for( int sp = 0; sp < desc.nproc_summa; sp++ ) {
      for( int i = 0; i < nBlocksPerProcRow; i++ ) {
	for( int rproc = 0; rproc < desc.nprocr; rproc++ ) {
	  for( int j = 0; j < nBlocksPerProcCol; j++ ) {
	    for( int cproc = 0; cproc < desc.nprocc; cproc++ ) {
	      int target = 0;
	      int source = cproc + rproc*desc.nprocc + sp*base;
	      // row and column of the beginning of the block in I
	      int row = j*(desc.nprocc*bs) + cproc*bs;
	      int col = i*(desc.nprocr*bs) + rproc*bs + sp*nBlocksPerBase*bs;
	      int offsetTarget = row + col*ldo;
	      int offsetSource = (j + i*nBlocksPerProcCol)*bs*bs;
	      receiveBlock( comm, rank, target, O+offsetTarget, bs, source, I+offsetSource, ldo );
	    }
	  }
	}
      }
    }
  } else { // recursively call on each of four submatrices
    desc.nrec -= 1;
    desc.lda /= 2;
    int entriesPerQuarter = numEntriesPerProc(desc);
    // top left
    collectTo1ProcRec( desc, O, I, ldo );
    // bottom left
    collectTo1ProcRec( desc, O+desc.lda, I + entriesPerQuarter, ldo );
    // top right
    collectTo1ProcRec( desc, O + desc.lda*ldo, I+2*entriesPerQuarter, ldo );
    // bottom right
    collectTo1ProcRec( desc, O + desc.lda*ldo+desc.lda, I+3*entriesPerQuarter, ldo );
  }
}

void collectTo1Proc( MatDescriptor desc, double *O, double *I ) {
  collectTo1ProcRec( desc, O, I, desc.lda );
}
