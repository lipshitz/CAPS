#include "generate.h"
#include "local-multiply.h"
#include "library.h"
#include <stdio.h>

int main( int argc, char **argv ) {

  int s = read_int( argc, argv, "-s", 10 );
  int n = read_int( argc, argv, "-n", 1<<s );
  int t = read_int( argc, argv, "-t", 1 );
  int S = read_int( argc, argv, "-S", s );
  int N = read_int( argc, argv, "-N", 1<<S );
  int d = read_int( argc, argv, "-d", 1 );

  double density = 1.*d/N;

  BlockIndexRowMajor rmi = BlockIndexRowMajor( 0, 0, n, n );
  BlockIndexColMajor cmi = BlockIndexColMajor( 0, 0, n, n );
  //Matrix *A = generateMatrix( &cmi, .002 );
  //Matrix *B = generateMatrix( &rmi, .002, time(0)+1 );
  vector<Matrix*> As;
  vector<Matrix*> Bs;
  for( int i = 0; i < t; i++ ) {
    As.push_back( generateMatrix( &cmi, density, time(0)+20*i ) );
    Bs.push_back( generateMatrix( &rmi, density, time(0)+20*i+10 ) );
  }

  double start_time = read_timer();
  for( int i = 0; i < t; i++ )
    sortDedup( local_multiply( As[i], Bs[i]), CompColMajorEntry );
  double stop_time = read_timer();
		  

  printf( "Total time %f, time per iteration %f, predicted time for full multiplication of size %d is %f\n",
	  stop_time-start_time, (stop_time-start_time)/t, N, (stop_time-start_time)/t*N/n*N/n*N/n );

}
