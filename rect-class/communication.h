#include <mpi.h>
#include <stdlib.h>

#ifndef COMMUNICATION_H
#define COMMUNICATION_H

int getRank();
void initCommunication( int *argc, char ***argv );
int getRelativeRank( int xOld, int xNew );
void reduceBy( int k, int x, double **args, double *out, int *n );
void expandBy( int k, int x, double **args, double *out, int *n );
void iReduceBy1( int k, int x, double **args, double *out, int* ns, MPI_Request *&req, double *&buffer );
void iReduceBy2( int k, int x, double **args, double *out, int* ns, MPI_Request *&req, double *&buffer );
void iExpandBy1( int k, int x, double **args, double *out, int* ns, MPI_Request *&req, double *&buffer );
void iExpandBy2( int k, int x, double **args, double *out, int* ns, MPI_Request *&req, double *&buffer );
void distFrom1ProcSq( double *A1, double *ADist, int n, int r, int P );
void distFrom1ProcTri( double *A1, double *ADist, int n, int r, int P );
void colTo1ProcSq( double *A1, double *ADist, int n, int r, int P );
void colTo1ProcTri( double *A1, double *ADist, int n, int r, int P );

#endif 
