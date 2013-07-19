#include "trsm.h"
#include "syrk.h"
#include "sizes.h"
#include "communication.h"
#include <stdlib.h>

#ifndef CHOL_H
#define CHOL_H

void chol( double *A, int n, int x, int r );
void cholBase( double *A, int n );
void cholDFS( double *A, int n, int x, int r );
void cholWaste2( double *A, int n, int x, int r );
void cholWaste2S( double *A, int n, int x, int r );

#endif
