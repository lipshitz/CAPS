#include "types.h"
#include <functional>

Matrix *merge( Entry* buf, int *counts, int *offsets, int P, std::function<bool(const pair<int,Entry>, const pair<int,Entry>)> reverseSort );
bool ReverseColMajor( pair<int,Entry> p1, pair<int,Entry> p2 );
bool ReverseRowMajor( pair<int,Entry> p1, pair<int,Entry> p2 );
