#ifndef MAT_DESCRIPTOR_H
#define MAT_DESCRIPTOR_H

#include <mpi.h>
#include <assert.h>
#include "library.h"

struct MatDescriptor {
  int lda;
  int nrec;
  int nproc;
  int nprocr;
  int nprocc;
  int nproc_summa;
  int bs;
};

const int DESC_SIZE = 7; // number of ints that make up a MatDescriptor

/*
  Matrix storage in hierarchical.
    At the top level, it is recursive-backwards N with nrec levels.
    Then it is 2d block cyclic with block size of  bs rows x bs columns.  
      There nproc will always be 7^k, and nprocr * nprocc = nproc.  These 
      specify the number of processors participating in any row or column
      of the submatrix.  The layout is column major, both within blocks and 
      between blocks. 
    nprocr is the number of processors that share any row of the block-cyclic
    layout, and similarly nprocc is the number of processors that share any column of it.
  It will always be assumed that lda is a multiple of 2^(nrec) * bs * nprocr and
  of 2^(nrec) * bs * nprocc
  Moreover, if nprocr != nprocc, it should always be that nprocr = SEVEN * nprocc


  Under these assumptions, the number of entries stored on each processor is 
  lda*lda/nproc.  The start points of the submatrices are:
  A11 = A
  A21 = A + lda*lda*nproc/4
  A12 = A + 2*lda*lda*nproc/4
  A22 = A + 3*lda*lda*nproc/4
  
  A descriptor is valid for a submatrix if its lda is divided by 2 and nrec
  is decreased by 1.  No other change is needed.
 */

void verifyDescriptor( MatDescriptor desc );
long long numEntriesPerProc( MatDescriptor desc );

#endif
