#include "rectsizes.h"
#include <stdio.h>
#include <stdlib.h>

int m0,n0,k0;
int Pmax;
int basemn, basemk, basenk;

void initSizesRect( int m, int n, int k, int P, int r, int *divPatt ) {
  Pmax = P;
  m0 = m;
  n0 = n;
  k0 = k;
  for( int i = 0; i < r; i++ ) {
    m0 /= divPatt[3*i];
    n0 /= divPatt[3*i+1];
    k0 /= divPatt[3*i+2];
  }
  basemn = (m0*n0+P-1)/P;
  basemk = (m0*k0+P-1)/P;
  basenk = (n0*k0+P-1)/P;
}

int getSizeRect( int m, int n, int k, int P ) {
  if( k == 1 ) {
    return basemn*(Pmax/P)*(m/m0)*(n/n0);
  } else if( n == 1 ) {
    return basemk*(Pmax/P)*(m/m0)*(k/k0);
  } else if( m == 1 ) {
    return basenk*(Pmax/P)*(n/n0)*(k/k0);
  } else {
    printf("Error: one of m,n,k must be 1 for a matrix\n %d %d %d %d", m,n,k,P);
    exit(-1);
  }
}
