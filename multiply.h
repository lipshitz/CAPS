#ifndef MULTIPLY_H
#define MULTIPLY_H

#include "dgemm-blas.h"
#include "summa1d.h"
#include "memory.h"
#include "matrix.h"
#include "counters.h"
#include "communication.h"
#include <assert.h>


void multiply( double *A, double *B, double *C, MatDescriptor desc, char* pattern=NULL );
void multiplyInternal( double *A, double *B, double *C, MatDescriptor desc, double *work );
void strassenBFS( double *A, double *B, double *C, MatDescriptor desc, double* work );
void strassenDFS( double *A, double *B, double *C, MatDescriptor desc, double *work );
void block_multiply( double *A, double *B, double *C, MatDescriptor desc, double *work );
void localStrassen4( double *A, double *B, double *C, MatDescriptor desc, double *work);
void strassenHybrid6( double *A, double *B, double *C, MatDescriptor desc, double *work);
void strassenHybrid5( double *A, double *B, double *C, MatDescriptor desc, double *work);
void strassenHybrid4( double *A, double *B, double *C, MatDescriptor desc, double *work);
void strassenHybrid3( double *A, double *B, double *C, MatDescriptor desc, double *work);
void strassenHybrid2( double *A, double *B, double *C, MatDescriptor desc, double *work);
void strassenHybrid1( double *A, double *B, double *C, MatDescriptor desc, double *work);
void strassen2DFS( double *A, double *B, double *C, MatDescriptor desc, double *work);

#endif
