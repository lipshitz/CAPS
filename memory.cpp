#include "memory.h"
#include <stdio.h>

long long availableMemory = 1024*1024*1024/4; // 2 GB, space for 2^28 doubles
double usedMemory = 0.;
double maxUsedMemory = 0.;
//long long usedMemory = 0;
//long long maxUsedMemory = 0;

double* allocate( long long s ) {
  usedMemory += s;
  maxUsedMemory = max(usedMemory,maxUsedMemory);
  return (double*) malloc( s*sizeof(double) );
}

void deallocate( double *A, long long s ) {
  usedMemory -= s;
  free(A);
}

// we need to figure out exactly what this does
bool enoughMemory( long long memNeeded ) {
  //return true; // does a BFS at every step, until we are down to 1 proc
  //  return false; // does a DFS at every step, until we need BFS to get to 1 proc

  // a real implementation
  //printf("availableMemory call need %lld *blowup %f have %lld/%lld\n", memNeeded, memNeeded*memoryBlowupFactor, availableMemory - usedMemory, availableMemory);
  return( availableMemory-usedMemory > memNeeded*memoryBlowupFactor );
}

void setAvailableMemory( long long am ) {
  availableMemory = am;
}

double getMaxUsedMemory() {
  return maxUsedMemory;
}
