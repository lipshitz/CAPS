#include "summa1d.h"
#include "command-line-parser.h"
#include "communication.h"
#include <math.h>

int main( int argc, char **argv ) {
  initCommunication( &argc, &argv );
  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  char *inFile = NULL;
  char *checkFile = NULL;
  MatDescriptor d;
  double *IA = NULL, *IB = NULL, *OC = NULL;

  if( rank == 0 ) {
    inFile = read_string( argc, argv, "-i", NULL );
    checkFile = read_string( argc, argv, "-c", NULL );
    d.bs = 1;
    
    if( !inFile ) {
      printf("Specify input file\n");
      MPI_Finalize();
      exit(-1);
    }
    FILE *f = fopen( inFile, "r" );
    if( !f ) {
      printf("Error opening file: %s\n", inFile );
      MPI_Finalize();
      exit(-1);
    }
    int reads = 0;
    reads += fscanf( f, "%d\n", &d.lda );
    IA = (double *) malloc( d.lda*d.lda*sizeof(double) );
    IB = (double *) malloc( d.lda*d.lda*sizeof(double) );
    OC = (double *) malloc( d.lda*d.lda*sizeof(double) );
    for( int i = 0; i < d.lda; i++ ) {
      for( int j = 0; j < d.lda; j++ )
        reads += fscanf( f, "%lf ", IA+i+j*d.lda );
      reads += fscanf(f, "\n");
    }
    for( int i = 0; i < d.lda; i++ ) {
      for( int j = 0; j < d.lda; j++ )
        reads += fscanf( f, "%lf ", IB+i+j*d.lda );
      reads += fscanf(f, "\n");
    }
    if( reads != d.lda*d.lda*2+1 ) {
      printf("Error reading file: %s, only read %d entries\n", inFile, reads);
      MPI_Finalize();
      exit(-1);      
    }
    printf("Read complete\n");
    d.nrec = 0;
    d.nproc = 1;
    d.nprocr = 1;
    d.nprocc = 1;
    MPI_Comm_size( MPI_COMM_WORLD, &d.nproc_summa );
  }
  assert( sizeof(MatDescriptor) == DESC_SIZE *sizeof(int) );
  if( rank != 0 || d.nproc_summa > 1 ) 
    MPI_Bcast( &d, DESC_SIZE, MPI_INT, 0, MPI_COMM_WORLD );

  // allocate A, B, C and distribute IA, IB
  double *A = (double*) malloc( d.lda*d.lda/d.nproc_summa*sizeof(double) );
  double *B = (double*) malloc( d.lda*d.lda/d.nproc_summa*sizeof(double) );
  double *C = (double*) malloc( d.lda*d.lda/d.nproc_summa*sizeof(double) );

  assert( d.lda % d.nproc_summa == 0 );
  int colbs = d.lda / d.nproc_summa;
  // put it block column layout
  MPI_Scatter( IA, d.lda*colbs, MPI_DOUBLE, A, d.lda*colbs, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Scatter( IB, d.lda*colbs, MPI_DOUBLE, B, d.lda*colbs, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  
  // run the multiply
  double *work = (double*) malloc( d.lda*colbs*sizeof(double) );
  double startTime = read_timer();
  MPI_Barrier(MPI_COMM_WORLD);
  summa1d( A, B, C, d, work );
  MPI_Barrier(MPI_COMM_WORLD);
  double endTime = read_timer();
  if( rank == 0 )
    printf("GFlop/s %f\n", 2.*d.lda*d.lda*d.lda/(endTime-startTime)/1.e9 );
  
  // gather the answer into OC
  MPI_Gather( C, d.lda*colbs, MPI_DOUBLE, OC, d.lda*colbs, MPI_DOUBLE, 0, MPI_COMM_WORLD );

  // output the result
  if( rank == 0 ) {
    if( checkFile ) {
      FILE *cFile = fopen(checkFile, "r");
      double CC;
      double error = 0;
      for( int i = 0; i < d.lda; i++ ) {
        for( int j = 0; j < d.lda; j++ ) {
          if( fscanf( cFile, "%lf ", &CC ) ) {
            double e = fabs(OC[i+j*d.lda]-CC);
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
      oFile = stdout;
      
      for( int i = 0; i < d.lda; i++ ) {
        for( int j = 0; j < d.lda; j++ )
          fprintf( oFile, "%3.0f ", OC[i+j*d.lda] );
        fprintf( oFile, "\n");
      }
    }
  }
  MPI_Finalize();
  
}
