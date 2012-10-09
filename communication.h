#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <mpi.h>
#include <cstring>
#include "matrix.h"
#include "library.h"
#include "counters.h"

void initCommunication(int *argc, char ***argv, int randomize = 0);
void initCommunicationAlt(int *argc, char ***argv, int randomize = 0);

// get the global rank, useful for printing
int getRank();
MPI_Comm getComm();
int getLog7nProcs();
int getPFactor();

// take 7 matrices distributed across 7^t processors as S1...S7 (contiguous), and rearrange them to be stored in 7^(t-1) processor, each processor only gets one
void gatherMatrices( MatDescriptor sDesc, double *S1234567, MatDescriptor tDesc, double *T, double *Work );
void gatherMatrices1( MatDescriptor sDesc, double *S0, double *S1, double *S2, int *SPos, MatDescriptor tDesc, double *Work, MPI_Request **reqs );
void gatherMatrices2( MatDescriptor sDesc, double *S0, double *S1, double *S2, double *S3, int *SPos, MatDescriptor tDesc, double *Work, MPI_Request *reqs );
void gatherMatrices3( MatDescriptor sDesc, MatDescriptor tDesc, double *Work, MPI_Request *reqs, double *T );
// inverse operation; take one matrix on each of 7 sets of 7^(t-1) processors and rearrange them to be all stored of all 7^t processors
void scatterMatrices( MatDescriptor sDesc, double *S, MatDescriptor tDesc, double *T1234567, double *Work );

void gatherMatricesS0( MatDescriptor sDesc, MatDescriptor tDesc, MPI_Request **reqs );
void gatherMatricesS1( MatDescriptor sDesc, double **Ss, int *SPos, int sNum, MatDescriptor tDesc, double *Work, MPI_Request *reqs );
void gatherMatricesS2( MPI_Request *reqs, MatDescriptor sDesc, MatDescriptor tDesc, double *Work, double *T );
void scatterMatricesS0( MatDescriptor sDesc, MatDescriptor tDesc, MPI_Request **reqs, double *T, double *work );
void scatterMatricesS1( MatDescriptor sDesc, double **Ss, int *SPos, int sNum, MatDescriptor tDesc, double *work, MPI_Request *reqs );
void scatterMatricesS2( MatDescriptor sDesc, MPI_Request *reqs );
// this is really a toy function for testing.  It takes a matrix all on 1 processor, and distributes it to match desc.  All processors must provide consistent descriptors.  However I may be NULL on processors with ranks other than 0 in desc; the processor with rank 0 is desc currently holds the matrix.
void distributeFrom1Proc( MatDescriptor desc, double *O, double *I );
void collectTo1Proc( MatDescriptor desc, double *O, double *I );

#endif
