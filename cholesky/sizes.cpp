#include "sizes.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// compute and store the sizes of matrices on various numbers of processors.

int *sizeSq, *sizeTri;
int r;

/*
void initSizes( int P, int r_in, int n ) {
  r = r_in;
  int logP = log(P+0.5)/log(2.);
  sizeSq = (int*) malloc( (logP+1)*(r+1)*sizeof(int) );
  sizeTri = (int*) malloc( (logP+1)*(r+1)*sizeof(int) );
  for( int l = 0; l <= logP; l++ ) {
    int nInt = n/(1<<r);    
    int p = 1<< l;
    sizeSq[l*r] = (nInt*nInt+p-1)/p;
    sizeTri[l*r] = (nInt*(nInt+1)/2+p-1)/p;
    for( int k = 1; k <= r; k++ ) {
      sizeSq[l*r+k] = sizeSq[l*r+k-1]*4;
      sizeTri[l*r+k] = 2*sizeTri[l*r+k-1]+sizeSq[l*r+k-1];
    }
  }
}
*/

void initSizes( int P, int r_in, int n ) {
  r = r_in;
  int logP = log(P+0.5)/log(2.);
  sizeSq = (int*) malloc( (logP+1)*(r+1)*sizeof(int) );
  sizeTri = (int*) malloc( (logP+1)*(r+1)*sizeof(int) );
  int nInt = n/(1<<r);
  sizeSq[logP*(r+1)] = (nInt*nInt+P-1)/P;
  sizeTri[logP*(r+1)] = (nInt*(nInt+1)/2+P-1)/P;
  for( int l = logP; l >= 0; l-- ) {
    if( l != logP ) {
      sizeSq[l*(r+1)] = 2*sizeSq[(l+1)*(r+1)];
      sizeTri[l*(r+1)] = 2*sizeTri[(l+1)*(r+1)];
    }
    for( int k = 1; k <= r; k++ ) {
      sizeSq[l*(r+1)+k] = 4*sizeSq[l*(r+1)+k-1];
      sizeTri[l*(r+1)+k] = 2*sizeTri[l*(r+1)+k-1]+sizeSq[l*(r+1)+k-1];
    }
  }
}

int getSizeSq( int r_in, int P ) {
  int logP = log(P+0.5)/log(2.);
  return sizeSq[logP*(r+1)+r_in];
}

int getSizeTri( int r_in, int P ) {
  int logP = log(P+0.5)/log(2.);
  return sizeTri[logP*(r+1)+r_in];
}
