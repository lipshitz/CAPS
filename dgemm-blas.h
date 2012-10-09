#ifndef DGEMM_BLAS_H
#define DGEMM_BLAS_H

extern "C" {
  // for computers with cblas
  //#include <cblas.h>
  // for calling fortran blas
  void dgemm_( char*, char*, int*,int*,int*, double*, double*,int*, double*,int*, double*, double*,int*);
  void daxpy_( int*, double*, double*, int*, double*, int* );
  void dcopy_( int*, double*, int*, double*, int* );
  void dscal_( int*, double*, double*, int* );
}

void square_dgemm_zero( int n, double *A, double *B, double *C );

#endif
