#include <cinttypes>
#include <vector>
#include <utility>

using namespace std;

// int_d is the integer type for matrix dimension indices
// for matrices of dimension up to 2^31
typedef int_least32_t int_d;
// for larger matrices
//typedef int_least64_t int_d;

// int_e is the integer type for indexing arrays of matrix entries
// for up to 32GB of memory per process (48GB if 64 bit ints are used as int_d)
typedef int_least32_t int_e;
// for more memory
//typedef int_least64_t int_e;

// int_m must be large enough to hold the number of entries (including zeros) in a local submatrix
typedef int_least64_t int_m;

// matrix blocks are stored like this:
typedef pair<int_d,int_d> Coord;
typedef pair<Coord,double> Entry;
typedef vector<Entry> Matrix;

