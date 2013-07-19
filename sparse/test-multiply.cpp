#include "generate.h"
#include "local-multiply.h"
#include <stdio.h>

int main( int argc, char **argv ) {

  //int n = 120000;
  //int m = 1000;
  int n = 2, m = 2;

  BlockIndexRowMajor rmi = BlockIndexRowMajor( 0, 0, n, m );
  BlockIndexColMajor cmi = BlockIndexColMajor( 0, 0, m, n );
  //Matrix *A = generateMatrix( &cmi, .002 );
  //Matrix *B = generateMatrix( &rmi, .002, time(0)+1 );
  Matrix *A = generateMatrix( &cmi, 1 );
  Matrix *B = generateMatrix( &rmi, 1, time(0)+1 );
  
  Matrix *C = sortDedup( local_multiply( A, B ), CompColMajorEntry );

  printf("%lu %lu %lu\n", A->size(), B->size(), C->size() );

  for( auto it = A->begin(); it != A->end(); it++ ) {
    printf("%d %d %f\n", it->first.first, it->first.second, it->second );
  }

  for( auto it = B->begin(); it != B->end(); it++ ) {
    printf("%d %d %f\n", it->first.first, it->first.second, it->second );
  }

  for( auto it = C->begin(); it != C->end(); it++ ) {
    printf("%d %d %f\n", it->first.first, it->first.second, it->second );
  }



}
