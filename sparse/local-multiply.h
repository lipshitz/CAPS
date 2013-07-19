#include "types.h"
#include <functional>
#include <cstddef>

/*
  local_multiply computes using an outer-product
  A should be in column-major order, and B should be in row-major order
  C is unordered, and unreduced
  It is up to the caller to delete C
  If the optional argument Cin is given, new entries are appended to it

  sortDedup sorts and reduces a matrix
  The output is ordered by the comparison passed in
  CompColMajor and CompRowMajor are good choices, but anything should work
  It is up to the caller to delete the output; by default the input is deleted
*/

#ifndef LOCAL_MULTIPLY_H
#define LOCAL_MULITPLY_H

Matrix *sortDedup( Entry *A, int_e sA, std::function<bool(const Entry, const Entry)> comp, bool del = true, vector<double> *times = NULL );
Matrix *sortDedup( Matrix *A, std::function<bool(const Entry, const Entry)> comp, bool del = true, vector<double> *times = NULL );
Matrix *local_multiply( Matrix *A, Matrix *B, Matrix *Cin = NULL );
Matrix *local_multiply( Entry *A, int_e sA, Entry *B, int_e sB, Matrix *Cin = NULL );
//bool CompColMajor( const Coord, const Coord );
//bool CompRowMajor( const Coord, const Coord );
bool CompColMajorEntry( const Entry c1, const Entry c2 );
bool CompRowMajorEntry( const Entry c1, const Entry c2 );

#endif
