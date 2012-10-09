#ifndef MEMORY_H
#define MEMORY_H

#include <stdlib.h>
#include "library.h"

// everything is in units of doubles
const double memoryBlowupFactor = 4.; // This seems to be right

double* allocate( long long s );
void deallocate( double *A, long long s );
bool enoughMemory( long long memNeeded );
void setAvailableMemory( long long am );
double getMaxUsedMemory();

#endif
