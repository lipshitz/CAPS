#include "matrix.h"

void verifyDescriptor( MatDescriptor desc ) {
  assert( desc.lda % ((1<<desc.nrec)*desc.bs*desc.nprocr) == 0 );
  assert( desc.lda % ((1<<desc.nrec)*desc.bs*desc.nprocc) == 0 );
  assert( desc.nprocr*desc.nprocc == desc.nproc );
}

long long numEntriesPerProc( MatDescriptor desc ) {
  long long lda = desc.lda;
  return ( (lda*lda) / desc.nproc / desc.nproc_summa );
}
