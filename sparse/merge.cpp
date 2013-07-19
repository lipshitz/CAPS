#include "types.h"
#include <functional>
#include <algorithm>

bool ReverseColMajor( pair<int,Entry> p1, pair<int,Entry> p2 ) {
  return ( (p1.second.first.second > p2.second.first.second) || ( p1.second.first.second == p2.second.first.second
								  && (p1.second.first.first > p2.second.first.first) ) );
}
bool ReverseRowMajor( pair<int,Entry> p1, pair<int,Entry> p2 ) {
  return ( (p1.second.first.first > p2.second.first.first) || ( p1.second.first.first == p2.second.first.first && 
								(p1.second.first.second > p2.second.first.second) ) );
}

Matrix *merge( Entry* buf, int *counts, int *offsets, int P, std::function<bool(const pair<int,Entry>, const pair<int,Entry>)> reverseSort ) {
  vector<pair<int,Entry> > next;
  int is[P];
  for( int i = 0; i < P; i++ )
    is[i] = 0;
  // initial population of the heap
  for( int i = 0; i < P; i++ )
    if( counts[i] > is[i] ) {
      next.push_back( make_pair( i,buf[offsets[i]+is[i]] ) );
      is[i]++;
    }
  make_heap(next.begin(), next.end(), reverseSort );

  Matrix* ret = new Matrix;
  // put the first entry in the matrix
  if( !next.empty() ) {
    pop_heap(next.begin(), next.end(), reverseSort );
    ret->push_back( next.back().second );
    int i = next.back().first;
    if( counts[i] > is[i] ) {
      next.back().second = buf[offsets[i]+is[i]];
      push_heap(next.begin(), next.end(), reverseSort );
      is[i]++;
    } else {
      next.pop_back();
    }
  }
    
  while( !next.empty() ) {
    pop_heap(next.begin(), next.end(), reverseSort );
    if( ret->back().first.first == next.back().second.first.first &&
	ret->back().first.second == next.back().second.first.second ) {
      ret->back().second += next.back().second.second;
    } else {
      ret->push_back( next.back().second );
    }
    int i = next.back().first;
    if( counts[i] > is[i] ) {
      next.back().second = buf[offsets[i]+is[i]];
      push_heap(next.begin(), next.end(), reverseSort );
      is[i]++;
    } else {
      next.pop_back();
    }
  }
  return ret;
}
