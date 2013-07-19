// pack a square matrix A of size rows x cols with layout lda into a backwards-N-recursive structure APacked with blocks of size bs.  Simultaneously pad up to size ldaPacked.
void packMatrix(int rows, int cols, int lda, double *A, int bs, int ldaPacked, double *APacked) {
  if( ldaPacked <= bs ) { // just copy the matrix, padding as necessary
#pragma omp parallel for schedule(static, (cols+NUM_THREADS-1)/NUM_THREADS)
    for( int c = 0; c < cols; c++ ) {
      memcpy( APacked+c*ldaPacked, A+c*lda, rows*sizeof(double) );
      memset( APacked+c*ldaPacked+rows, 0, (ldaPacked-rows)*sizeof(double) );
    }
    for( int c = cols; c < ldaPacked; c++ )
      memset( APacked+c*ldaPacked, 0, ldaPacked*sizeof(double) );
  } else { // recursively call packMatrix on each of the four submatrices
    // top left
    //assert(ldaPacked % 2 == 0);
    int halfsize = ldaPacked/2;
    packMatrix( halfsize, halfsize, lda, A, bs, ldaPacked/2, APacked );
    // bottom left
    packMatrix( rows-halfsize, halfsize, lda, A+(halfsize), bs, ldaPacked/2, APacked+ldaPacked*ldaPacked/4 );
    // top right
    packMatrix( halfsize, cols-halfsize, lda, A+(halfsize)*lda, bs, ldaPacked/2, APacked+ldaPacked*ldaPacked/2 );
    // bottom right
    packMatrix( rows-halfsize, cols-halfsize, lda, A+(halfsize)*lda+(halfsize), bs, ldaPacked/2, APacked+ldaPacked*ldaPacked*3/4 );
  }
}

void unpackMatrix(int rows, int cols, int lda, double *A, int bs, int ldaPacked, double *APacked) {
  if( ldaPacked <= bs ) { // just copy the matrix, padding as necessary
    for( int c = 0; c < cols; c++ )
      memcpy( A+c*lda, APacked+c*(ldaPacked), rows*sizeof(double) );
  } else { // recursively call unpackMatrix on each of the four submatrices
    // top left
    int halfrows = (ldaPacked)/2;
    int halfcols = (ldaPacked)/2;
    unpackMatrix( halfrows, halfcols, lda, A, bs, ldaPacked/2, APacked);
    // bottom left
    unpackMatrix( rows-halfrows, halfcols, lda, A+(halfrows), bs, ldaPacked/2, APacked+ldaPacked*ldaPacked/4);
    // top right
    unpackMatrix( halfrows, cols-halfcols, lda, A+(halfcols)*lda, bs, ldaPacked/2, APacked+ldaPacked*ldaPacked/2);
    // bottom right
    unpackMatrix( rows-halfrows, cols-halfcols, lda, A+(halfcols)*lda+(halfrows), bs, ldaPacked/2, APacked+ldaPacked*ldaPacked*3/4 );
  }
}

void pack( int n, double *A, double *Ap, int bs ) {
  packMatrix( n, n, n, A, bs, n, Ap );
}

void unpack( int n, double *A, double *Ap, int bs ) {
  unpackMatrix( n, n, n, A, bs, n, Ap ); 
}

