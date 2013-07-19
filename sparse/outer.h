#include "types.h"
#include "library.h"

/*
  Does parallel matrix multiplication using the outer product algorithm
  Each processor should pass in a block column of A in column-major layout
  and the corresponding block row of B in row-major layout.
  The output is distributed by block columns in column-major layout.
  It is up to the calling function to delete the output.  
 */
Matrix *outerProduct( Matrix *A, Matrix *B, int_d n, int P, vector<double> *times = NULL );
