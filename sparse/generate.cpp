#include <boost/random.hpp>
#include "generate.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <set>

Matrix *generateMatrix( SubMatrixIndex *indexer, double p, int seed) {
  Matrix *ret = new Matrix();
  int_m nEntries = indexer->getSize();
  
  typedef boost::mt19937 RNGType;
  RNGType rng(seed);


  boost::binomial_distribution<int_m> bin( nEntries, p );
  boost::variate_generator< RNGType, boost::binomial_distribution<int_m> > binom( rng, bin );

  int_e nnz = binom();

  std::set<int_m> chosenIndices;
  boost::uniform_int<int_m> uniform_entry( 0, nEntries-1 );
  boost::variate_generator< RNGType, boost::uniform_int<int_m> > random_entry(rng, uniform_entry);

  while( chosenIndices.size() < nnz )
    chosenIndices.insert( random_entry() );

  boost::uniform_01<double,double> uniform_value;
  boost::variate_generator< RNGType, boost::uniform_01<double,double> > random_value(rng, uniform_value);

  for( std::set<int_m>::iterator it = chosenIndices.begin(); it != chosenIndices.end(); it++ ) {
    int_d i,j;
    indexer->getPosition( *it, i, j );
    ret->push_back( std::make_pair( std::make_pair(i,j), random_value() ) );
  }
  return ret;
}
