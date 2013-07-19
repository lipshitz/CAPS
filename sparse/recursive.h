#include "types.h"
#include <mpi.h>

MPI_Comm *initCommunication( int r, int *patt );
Matrix *recursiveMultiply( Matrix *A, Matrix *B, int log2n, vector<double> *times = NULL );
pair<Entry*,int_e> recMM( Entry *A, int_e sA, Entry *B, int_e sB, int_d m, int log2m, int_d mStartA, int_d mStartB, int recLevel );
pair<Entry*,int_e> splitC( Entry *A, int_e sA, Entry *B, int_e sB, int_d m, int log2m, int_d mStartA, int_d mStartB, int recLevel );
pair<Entry*,int_e> splitAB( Entry *A, int_e sA, Entry *B, int_e sB, int_d m, int log2m, int_d mStartA, int_d mStartB, int recLevel );
