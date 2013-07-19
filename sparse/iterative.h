#include "types.h"
#include <mpi.h>

Matrix *iterative3D( Matrix *A, Matrix *B, int_d n, int c, int sqrtPoc, int sqrtc, MPI_Comm rowComm, int rowRank, MPI_Comm colComm, int colRank, MPI_Comm fibComm, int fibRank, vector<double> *times = NULL );

