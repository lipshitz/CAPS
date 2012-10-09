#include "summa1d.h"
#include "communication.h"

#define TAG_SUMMA 100

// assumes the matrix is in block-column layout.
void summa1d( double *A, double *B, double *C, MatDescriptor desc, double *work ) {
#ifdef SANITY_CHECKS
  assert( desc.nrec == 0 );
  assert( desc.nprocr == 1 );
  assert( desc.nprocc == 1 );
  assert( desc.nproc == 1 );
  assert( desc.bs == 1 ); // otherwise it probably isn't in block-column layout
  // desc.nproc_summa will specify how many processors are used here.
  assert( desc.lda % desc.nproc_summa == 0 );
#endif
  MPI_Comm comm = getSummaComm(desc);
  int rank;
  MPI_Comm_rank( comm, &rank );
  int colBlockSize = desc.lda / desc.nproc_summa;
  char N = 'N';
  double one = 1.;
  double zero = 0.;
  long long lda = desc.lda;
  long long lda3 = lda*lda*lda;
  increaseAdditions(lda3/desc.nproc_summa);
  increaseMultiplications(lda3/desc.nproc_summa);
  increaseMessages(2*(desc.nproc_summa-1));
  increaseWords(2*(desc.nproc_summa-1)*desc.lda*colBlockSize*sizeof(double));

  double *current, *next;
  int messageSize = desc.lda*colBlockSize;
  next = A;
  MPI_Request sendReq, recvReq;
  for( int offset = 0; offset < desc.nproc_summa; offset++ ) {
    int k = (rank + desc.nproc_summa - offset) % desc.nproc_summa;
    // do the necessary communication for the next step
    startTimer(TIMER_COMM_SUMMA);
    if( offset > 0 ) {
      // make sure the last communication steps are finished
      MPI_Wait( &sendReq, MPI_STATUS_IGNORE );
      MPI_Wait( &recvReq, MPI_STATUS_IGNORE );
    }
    // update current and next appropriately
    current = next;
    if( current != work )
      next = work;
    else
      next = work+messageSize;
    if( offset < desc.nproc_summa - 1 ) {
      MPI_Isend( current, messageSize, MPI_DOUBLE, (rank+1)%desc.nproc_summa, TAG_SUMMA, comm, &sendReq );
      MPI_Irecv( next, messageSize, MPI_DOUBLE, (rank+desc.nproc_summa-1) % desc.nproc_summa, TAG_SUMMA, comm, &recvReq );
    }
    stopTimer(TIMER_COMM_SUMMA);

    double *Bkj = B+k*colBlockSize;
    // current holds the k'th column block of A
    // multiply:
    startTimer(TIMER_MUL);
    if( offset == 0 )
      dgemm_( &N, &N, &desc.lda, &colBlockSize, &colBlockSize, &one, current, &desc.lda, Bkj, &desc.lda, &zero, C, &desc.lda );
    else
      dgemm_( &N, &N, &desc.lda, &colBlockSize, &colBlockSize, &one, current, &desc.lda, Bkj, &desc.lda, &one, C, &desc.lda );
    stopTimer(TIMER_MUL);
  }
}

