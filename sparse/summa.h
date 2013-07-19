#include "types.h"
#include <mpi.h>

Matrix *spSUMMA( Matrix *A, Matrix *B, int_d n, int sqrtP, MPI_Comm rowComm, int rowRank, MPI_Comm colComm, int colRank, vector<double> *times = NULL );
