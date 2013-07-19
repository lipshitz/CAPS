#include "blas.h"

#ifndef TRSM_H
#define TRSM_H

void trsm( double *X, double *T, int n, int x, int r );
void trsmBase( double *X, double *T, int n );
void trsmWasteX( double *X, double *T, int n, int x );
void trsmDFS( double *X, double *T, int n, int x, int r );
void trsmBFS( double *X, double *T, int n, int x, int r );

#endif 
