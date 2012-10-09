#include "dgemm-blas.h"
#include <omp.h>

void square_dgemm_zero( int n, double *A, double *B, double *C ) {
  // for cblas
  //cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, n,n,n, 1, A,n, B,n, 0, C,n );

  // for fortran blas
  char N = 'N';
  double one = 1.;
  double zero = 0.;
  dgemm_( &N, &N, &n,&n,&n, &one, A,&n, B,&n, &zero, C,&n);
}
