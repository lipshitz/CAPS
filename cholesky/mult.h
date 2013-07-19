#include "blas.h"

#ifndef MULT_H
#define MULT_H

void mult( double *C, double *A, double *B, int n, int x, int r, double alpha=1. );
void multBase( double *C, double *A, double *B, int n, int r, double alpha=1. );
void multWasteX( double *C, double *A, double *B, int n, int x, double alpha=1. );
void multDFS( double *C, double *A, double *B, int n, int x, int r, double alpha=1. );
void multBFS8( double *C, double *A, double *B, int n, int x, int r, double alpha=1. );
void multBFS4( double *C, double *A, double *B, int n, int x, int r, double alpha=1. );
void multBFS2( double *C, double *A, double *B, int n, int x, int r, double alpha=1. );

#endif
