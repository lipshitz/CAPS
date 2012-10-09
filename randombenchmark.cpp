/*
 * This version creates random distributed matrices and times
 * multiplication.  It does not check for accuracy, as that
 * would add quite a lot of runtime.  Instead, use eg fromfile1 to 
 * verify that multiply() functions correctly.
 */

#include "communication.h"
#include "library.h"
#include "multiply.h"
#include "memory.h"
#include "counters.h"
#include "command-line-parser.h"
#include <math.h>
#include <mpi.h>

const int MIN_STRASSEN = 512;

void fill( double *p, int n ) {
  for( int i = 0; i < n; i++ )
    p[i] = 2*drand48() - 1;
}

int main( int argc, char **argv ) {
  int randomize = read_int( argc, argv, "--random", 0 );
  initCommunicationAlt(&argc, &argv, randomize);
  initTimers();

  int rank = getRank();

  MatDescriptor desc;
  double *A, *B, *C;

  //if( rank == 0 ) { // decide what desc should be
  int size = read_int( argc, argv, "-s", -1 );
  if( size == -1 ) {
    if( rank == 0 )
      printf("Must specify the matrix size desired with -s\n");
    MPI_Finalize();
    exit(-1);      
  }
  
  int bsReq = read_int( argc, argv, "-b", 1 );
  int nrecReq = read_int( argc, argv, "-r", -1 );
  int mem = read_int( argc, argv, "-m", 1536 ); // number of megabytes available
  char *pattern = read_string( argc, argv, "-p", NULL );
  mem = read_int( argc, argv, "-k", mem*1024 ); // or kilobytes
  setAvailableMemory(mem*128); // convert kilobytes to doubles
  
  int log7nProcs = getLog7nProcs();
  if( nrecReq == -1 ) {
    int rsize = size / MIN_STRASSEN;
    nrecReq = 0;
    while( rsize > 1 ) {
      nrecReq += 1;
      rsize /= 2;
    }
    nrecReq = max(log7nProcs, nrecReq);
    if( getRank() == 0 )
      printf("Setting nrec=%d\n", nrecReq);
  }
  if( getRank() == 0 )
    printf("Benchmarking size %d\n", size);
  desc.lda = size;
  desc.nrec = nrecReq;
  desc.bs = bsReq;
  desc.nprocr = 1;
  desc.nprocc = 1;
  while( log7nProcs >= 2 ) {
    log7nProcs -= 2;
    desc.nprocr *= SEVEN;
    desc.nprocc *= SEVEN;
  }
  if( log7nProcs == 1 )
    desc.nprocr *= SEVEN;
  desc.nproc = desc.nprocr*desc.nprocc;
  desc.nproc_summa = getPFactor();

  // allocate the initial matrices
  A = allocate( numEntriesPerProc(desc) );
  B = allocate( numEntriesPerProc(desc) );
  C = allocate( numEntriesPerProc(desc) );

  // fill the matrices with random data
  fill( A, numEntriesPerProc(desc) );
  fill( B, numEntriesPerProc(desc) );


  MPI_Barrier( MPI_COMM_WORLD );
  double startTime = read_timer();
  multiply( A, B, C, desc, pattern );
  MPI_Barrier( MPI_COMM_WORLD );
  double endTime = read_timer();

  deallocate( A, numEntriesPerProc(desc) );
  deallocate( B, numEntriesPerProc(desc) );

  printCounters(endTime-startTime, desc);
  MPI_Finalize();
}
