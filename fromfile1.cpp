#include "communication.h"
#include "library.h"
#include "multiply.h"
#include "memory.h"
#include "counters.h"
#include "command-line-parser.h"
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

int main( int argc, char **argv ) {
  bool randomize = read_int( argc, argv, "--random", 0 );
  initCommunication(&argc, &argv, randomize);
  initTimers();

  int rank = getRank();

  MatDescriptor desc;
  double *A, *B, *IA=NULL, *IB=NULL, *C, *OC=NULL;
  char *outFile = NULL;
  char *checkFile = NULL;

  int mem = read_int( argc, argv, "-m", 1536 ); // number of megabytes available
  mem = read_int( argc, argv, "-k", mem*1024 ); // or kilobytes
  setAvailableMemory(mem*128); // convert kilobytes to doubles
  if( rank == 0 ) { // read parameters, the matrices, decide what desc should be
    char *inFile = read_string( argc, argv, "-i", NULL );
    outFile = read_string( argc, argv, "-o", NULL );
    checkFile = read_string( argc, argv, "-c", NULL );
    int bsReq = read_int( argc, argv, "-b", -1 );
    int nrecReq = read_int( argc, argv, "-r", -1 );

    if( !inFile ) {
      printf("Specify input file\n");
      MPI_Finalize();
      exit(-1);
    }
    FILE *f = fopen( inFile, "r" );
    if( !f ) {
      printf("Error opening file: %s\n", inFile);
      MPI_Finalize();
      exit(-1);
    }
    int reads = 0;
    reads += fscanf( f, "%d\n", &desc.lda );
    IA = allocate( desc.lda*desc.lda );
    IB = allocate( desc.lda*desc.lda );
    for( int i = 0; i < desc.lda; i++ ) {
      for( int j = 0; j < desc.lda; j++ )
	reads += fscanf( f, "%lf ", IA+i+j*desc.lda );
      reads += fscanf(f, "\n");
    }
    for( int i = 0; i < desc.lda; i++ ) {
      for( int j = 0; j < desc.lda; j++ )
	reads += fscanf( f, "%lf ", IB+i+j*desc.lda );
      reads += fscanf(f, "\n");
    }
    if( reads != desc.lda*desc.lda*2+1 ) {
      printf("Error reading file: %s, only read %d entries\n", inFile, reads);
      MPI_Finalize();
      exit(-1);      
    }
    printf("Read complete\n");
    // determine appropriate parameters

    if( bsReq == -1 || nrecReq == -1 ) {
      printf("Specify parameters with -b and -r.  Automatic determination of parameters not yet available\n");
      MPI_Finalize();
      exit(-1);      
    }
    desc.nrec = nrecReq;
    desc.bs = bsReq;
    int log7nProcs = getLog7nProcs();
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
  }
  // broadcast the descriptor to everyone, then distribute the matrices
  assert( sizeof(MatDescriptor) == DESC_SIZE *sizeof(int) );
  int nproc;
  MPI_Comm_size( MPI_COMM_WORLD, &nproc );
  if( nproc > 1 ) {
    MPI_Bcast( &desc, DESC_SIZE, MPI_INT, 0, getComm() );
  }
  A = allocate( numEntriesPerProc(desc) );
  B = allocate( numEntriesPerProc(desc) );
  distributeFrom1Proc( desc, A, IA );
  distributeFrom1Proc( desc, B, IB );

  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 ) {
    deallocate( IA, desc.lda*desc.lda );
    deallocate( IB, desc.lda*desc.lda );
  }
  C = allocate( numEntriesPerProc(desc ) );
  char *pattern = read_string( argc, argv, "-p", NULL );
  MPI_Barrier( MPI_COMM_WORLD );

  double startTime = read_timer();
  multiply( A, B, C, desc, pattern );
  MPI_Barrier( MPI_COMM_WORLD );
  double endTime = read_timer();

  deallocate( A, numEntriesPerProc(desc) );
  deallocate( B, numEntriesPerProc(desc) );

  if( rank == 0 ) {
    OC = allocate( desc.lda*desc.lda );
  }

  collectTo1Proc( desc, OC, C );


  // output the result
  if( rank == 0 ) {
    if( checkFile ) {
      FILE *cFile = fopen(checkFile, "r");
      double CC;
      double error = 0;
      for( int i = 0; i < desc.lda; i++ ) {
	for( int j = 0; j < desc.lda; j++ ) {
	  if( fscanf( cFile, "%lf ", &CC ) ) {
	    double e = fabs(OC[i+j*desc.lda]-CC);
	    if( e > error )
	      error = e;
	  }
	  else {
	    printf("Error reading verification file\n");
	    exit(-1);
	  }
	}
	if( fscanf(cFile, "\n") ) {
	    printf("Error reading verification file\n");
	    exit(-1);
	}
      }
      printf("Maximum deviation %e\n", error);
    } else {
      FILE *oFile;
      if( outFile )
	oFile = fopen( outFile, "w" );
      else
	oFile = stdout;
      
      for( int i = 0; i < desc.lda; i++ ) {
	for( int j = 0; j < desc.lda; j++ )
	  fprintf( oFile, "%.18le ", OC[i+j*desc.lda] );
	fprintf( oFile, "\n");
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  printCounters(endTime-startTime, desc);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
