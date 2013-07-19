#include "generate.h"
#include <stdio.h>

int main( int argc, char **argv ) {

  BlockIndexRowMajor indexer = BlockIndexRowMajor( 0, 0, 10, 10 );
  Matrix *data = generateMatrix( &indexer, .1 );

  for( auto it = data->begin(); it != data->end(); it++ ) {
    printf("%d %d %f\n", it->first.first, it->first.second, it->second );
  }

}
