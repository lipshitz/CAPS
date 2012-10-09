#define max(a,b) (((a)>(b))?(a):(b))
#ifndef LIBRARY_H
#define LIBRARY_H

// Although 7 is the only number that makes sense as far as doing strassen, the communication code can be tested more easily by setting this to a lower value.  Increasing it should give the communication code for other matrix multiplication algorithms, although the algorithms themselves would need to be written
const int SEVEN = 7;

double read_timer();

#endif
