#include "matrix.h"
#include "library.h"
#include "dgemm-blas.h"
#include "counters.h"
#include <cstring>

void summa1d( double *A, double *B, double *C, MatDescriptor desc, double *work );
extern MPI_Comm getSummaComm( MatDescriptor desc );
