#ifndef GENERATE_H
#define GENERATE_H
#include "types.h"
#include <vector>
#include <tuple>

#include <boost/random.hpp>



using namespace std;

class SubMatrixIndex {
 public:
  virtual void getPosition( int_m index, int_d &i, int_d &j ) = 0;
  virtual int_m getSize() = 0;
  virtual ~SubMatrixIndex() {};
};

class BlockIndexColMajor : public SubMatrixIndex {
 public:
 BlockIndexColMajor( int_d i_start, int_d j_start, int_d i_size, int_d j_size ) : 
  i_start( i_start ),  j_start( j_start ), i_size( i_size ), j_size( j_size ) {}
  virtual void getPosition( int_m index, int_d &i, int_d &j ) {
    i = index % i_size;
    j = index / i_size;
    i += i_start;
    j += j_start;
  }
  virtual int_m getSize() { return ((int_m)i_size)*j_size; }

 private:
  int_d i_start, j_start, i_size, j_size;
};

class BlockIndexRowMajor : public SubMatrixIndex {
 public:
 BlockIndexRowMajor( int_d i_start, int_d j_start, int_d i_size, int_d j_size ) : 
  i_start( i_start ),  j_start( j_start ), i_size( i_size ), j_size( j_size ) {}
  virtual void getPosition( int_m index, int_d &i, int_d &j ) {
    i = index / j_size;
    j = index % j_size;
    i += i_start;
    j += j_start;
  }
  virtual int_m getSize() { return ((int_m)i_size)*j_size; }

 private:
  int_d i_start, j_start, i_size, j_size;
};

// generate part of a random matrix, described by indexer, where each entry is nonzero with probability p.  This is optimized for p small; for p close to one a different generator should be used
Matrix *generateMatrix( SubMatrixIndex *indexer, double p, int seed = time(0) );

#endif
