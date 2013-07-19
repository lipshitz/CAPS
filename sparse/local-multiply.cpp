#include "types.h"
#include "library.h"
#include <map>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <algorithm>

bool CompColMajorEntry( const Entry c1, const Entry c2 ) {
  return ( (c1.first.second < c2.first.second) || ( c1.first.second == c2.first.second && (c1.first.first < c2.first.first) ) );
}
bool CompRowMajorEntry( const Entry c1, const Entry c2 ) {
  return ( (c1.first.first < c2.first.first) || ( c1.first.first == c2.first.first && (c1.first.second < c2.first.second) ) );
}

Matrix *sortDedup( Matrix *A, std::function<bool(const Entry, const Entry)> comp, bool del, vector<double> *times) {
  // sort and combine entries
  double t1 = read_timer();
  std::sort(A->begin(),A->end(),comp);
  double t2 = read_timer();
  Matrix *ret = new Matrix();
  if( !A->empty() ) {
    ret->push_back(*(A->begin()));
    for( auto it = A->begin()+1; it != A->end(); it++ ) {
      auto b = ret->end()-1;
      if( it->first.first == b->first.first && it->first.second == b->first.second )
	b->second += it->second;
      else
	ret->push_back(*it);
    }
  }
  double t3 = read_timer();

  if( del )
    delete A;

  if( times ) {
    times->push_back(t2-t1); // sort
    times->push_back(t3-t2); // reduce
  }
  return ret;
}

Matrix *sortDedup( Entry *A, int_e sA, std::function<bool(const Entry, const Entry)> comp, bool del, vector<double> *times) {
  // sort and combine entries
  double t1 = read_timer();
  std::sort(A,A+sA,comp);
  double t2 = read_timer();
  Matrix *ret = new Matrix();
  if( sA ) {
    ret->push_back(A[0]);
    for( auto it = A+1; it != A+sA; it++ ) {
      auto b = ret->end()-1;
      if( it->first.first == b->first.first && it->first.second == b->first.second )
	b->second += it->second;
      else
	ret->push_back(*it);
    }
  }
  double t3 = read_timer();

  if( del )
    delete[] A;

  if( times ) {
    times->push_back(t2-t1); // sort
    times->push_back(t3-t2); // reduce
  }
  return ret;
}

Matrix *local_multiply( Matrix *A, Matrix *B, Matrix *Cin ) {

  Matrix *ret = Cin;
  if( !ret ) {
    ret = new Matrix;
  }

  auto itA = A->begin();
  auto itB = B->begin();
  
  while( itA != A->end() && itB != B->end() ) {
    
    // advance in A until the column of A is at least the row of B
    while( (itA != A->end()) && (itA->first.second < itB->first.first) ) {
      itA++;
    }
    // advance in B until the row of B is at least the column of A
    while( (itB != B->end()) && (itB->first.first < itA->first.second) ) {
      itB++;
    }    

    // do an outer product, if these is one to do
    if( itA != A->end() && itB != B->end() && itA->first.second == itB->first.first ) {
      auto iA = itA, iB = itB;
      for( iA = itA; iA != A->end() && iA->first.second == itA->first.second; iA++ )
	for( iB = itB; iB != B->end() && iB->first.first == itB->first.first; iB++ ) {
	  Coord c = make_pair( iA->first.first, iB->first.second );
	  double v = iA->second * iB->second;
	  ret->push_back( make_pair(c,v) );
	}
      // advance past these
      itA = iA;
      itB = iB;
    }
  }

  return ret;
}

Matrix *local_multiply( Entry *A, int_e sA, Entry *B, int_e sB, Matrix *Cin ) {

  Matrix *ret = Cin;
  if( !ret ) {
    ret = new Matrix;
  }

  auto itA = A;
  auto itB = B;
  
  while( itA != A+sA && itB != B+sB ) {
    
    // advance in A until the column of A is at least the row of B
    while( (itA != A+sA) && (itA->first.second < itB->first.first) ) {
      itA++;
    }
    // advance in B until the row of B is at least the column of A
    while( (itB != B+sB) && (itB->first.first < itA->first.second) ) {
      itB++;
    }    

    // do an outer product, if these is one to do
    if( itA != A+sA && itB != B+sB && itA->first.second == itB->first.first ) {
      auto iA = itA, iB = itB;
      for( iA = itA; iA != A+sA && iA->first.second == itA->first.second; iA++ )
	for( iB = itB; iB != B+sB && iB->first.first == itB->first.first; iB++ ) {
	  Coord c = make_pair( iA->first.first, iB->first.second );
	  double v = iA->second * iB->second;
	  ret->push_back( make_pair(c,v) );
	}
      // advance past these
      itA = iA;
      itB = iB;
    }
  }

  return ret;
}

bool CompColMajor( const Coord c1, const Coord c2 ) {
  return ( (c1.second < c2.second) || ( c1.second == c2.second && (c1.first < c2.first) ) );
}

bool CompRowMajor( const Coord c1, const Coord c2 ) {
  return ( (c1.first < c2.first) || ( c1.first == c2.first && (c1.second < c2.second) ) );
}

