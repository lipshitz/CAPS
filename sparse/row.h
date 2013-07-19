#include "types.h"
#include "library.h"

/* 
   Compute using the (Naive) Block Row algorithm
   A and B are both distributed in block rows.  In each block,
   A is in column-major order and B is in row-major order.
   The output is distributed in block rows, in column-major order.
 */

Matrix *blockRow( Matrix *A, Matrix *B, int_d n, int rank, int P, vector<double> *times = NULL );
