#include "mult.h"
#include <stdlib.h>
#include "sizes.h"
#include "communication.h"

#ifndef SYRK_H
#define SYRK_H

void syrk( double *C, double *A, int n, int x, int r, double alpha=1. );
void syrkWasteX( double *C, double *A, int n, int x, double alpha=1. );
void syrkDFS( double *C, double *A, int n, int x, int r, double alpha=1. );
void syrkBFS8( double *C, double *A, int n, int x, int r, double alpha=1. );
void syrkBFS4( double *C, double *A, int n, int x, int r, double alpha=1. );
void syrkBFS2( double *C, double *A, int n, int x, int r, double alpha=1. );
void syrkBase( double *C, double *A, int n, double alpha=1. );

#endif
