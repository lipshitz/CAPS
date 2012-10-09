#include "communication.h"
#include "library.h"

int main( int argc, char **argv ) {
  initCommunication(&argc, &argv);
 
  MPI_Barrier( MPI_COMM_WORLD );
  // create a SEVEN*SEVEN x SEVEN*SEVEN matrix, and distribute with r=0,b=1.  Assumes we run on SEVEN^4 procs
  MatDescriptor desc;
  desc.lda = SEVEN*SEVEN;
  desc.nrec = 0;
  desc.nproc = SEVEN*SEVEN*SEVEN*SEVEN;
  desc.nprocr = SEVEN*SEVEN;
  desc.nprocc = SEVEN*SEVEN;
  desc.bs = 1;
  double I[SEVEN*SEVEN*SEVEN*SEVEN];
  int rank = getRank();
  if( rank == 0 ) {
    printf("Initial matrix\n");
    for( int i = 0; i < SEVEN*SEVEN*SEVEN*SEVEN; i++ )
      I[i] = i;
    for( int i = 0; i < SEVEN*SEVEN; i++ ) {
      for( int j = 0; j < SEVEN*SEVEN; j++ )
	printf("%.0f ", I[i+j*SEVEN*SEVEN]);
      printf("\n");
    }
  }

  double O[numEntriesPerProc(desc)*SEVEN*SEVEN*SEVEN];
  distributeFrom1Proc( desc, O, I );
  for( int p = 0; p < desc.nproc; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < numEntriesPerProc(desc); i++ )
	printf("%.0f ", O[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }

  for( int i = numEntriesPerProc(desc); i < numEntriesPerProc(desc)*SEVEN*SEVEN*SEVEN; i++ )
    O[i] = 0;
  // now call gathermatrices:
  
  MatDescriptor desc2 = desc;
  desc2.nproc /= SEVEN;
  desc2.nprocc /= SEVEN;
  
  double work[numEntriesPerProc(desc)*SEVEN];
  double O2[numEntriesPerProc(desc)*SEVEN*SEVEN*SEVEN];
  
  gatherMatrices( desc, O, desc2, O2, work );

  for( int p = 0; p < desc.nproc; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < numEntriesPerProc(desc2); i++ )
	printf("%.0f ", O2[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  for( int i = numEntriesPerProc(desc)*SEVEN; i < numEntriesPerProc(desc)*SEVEN*SEVEN*SEVEN; i++ )
    O2[i] = 0;

  // gather again
  MatDescriptor desc3 = desc2;
  desc3.nproc /= SEVEN;
  desc3.nprocr /= SEVEN;
  double work2[numEntriesPerProc(desc2)*SEVEN*SEVEN];
  double O22[numEntriesPerProc(desc2)*SEVEN*SEVEN];
  gatherMatrices( desc2, O2, desc3, O22, work2 );

  for( int p = 0; p < desc.nproc; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < numEntriesPerProc(desc3); i++ )
	printf("%.0f ", O22[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  for( int i = numEntriesPerProc(desc2)*SEVEN; i < numEntriesPerProc(desc2)*SEVEN*SEVEN; i++ )
    O22[i] = 0;

  // gather again
  MatDescriptor desc4 = desc3;
  desc4.nproc /= SEVEN;
  desc4.nprocc /= SEVEN;
  double work3[numEntriesPerProc(desc2)*SEVEN*SEVEN];
  double O23[numEntriesPerProc(desc4)*SEVEN*SEVEN];
  gatherMatrices( desc3, O22, desc4, O23, work3 );

  for( int p = 0; p < desc.nproc; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < numEntriesPerProc(desc4); i++ )
	printf("%.0f ", O23[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  for( int i = numEntriesPerProc(desc4); i < numEntriesPerProc(desc4)*SEVEN*SEVEN; i++ )
    O23[i] = 0;

  // gather again
  MatDescriptor desc5 = desc4;
  desc5.nproc /= SEVEN;
  desc5.nprocr /= SEVEN;
  double work4[numEntriesPerProc(desc2)*SEVEN*SEVEN*SEVEN];
  double O24[numEntriesPerProc(desc2)*SEVEN*SEVEN*SEVEN];
  gatherMatrices( desc4, O23, desc5, O24, work4 );

  for( int p = 0; p < desc.nproc; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < numEntriesPerProc(desc5); i++ )
	printf("%.0f ", O24[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  /*
  //create a 4*SEVEN^2 x 4*SEVEN^2 matrix and distribute it with r=2,b=1.  Assumes we run on SEVEN^3 procs
  MatDescriptor desc;
  desc.lda = 4*SEVEN*SEVEN;
  desc.nrec = 2;
  desc.nproc = SEVEN*SEVEN*SEVEN;
  desc.nprocr = SEVEN*SEVEN;
  desc.nprocc = SEVEN;
  desc.bs = 1;
  double I[4*4*SEVEN*SEVEN*SEVEN*SEVEN];
  int rank = getRank();
  if( rank == 0 ) {
    printf("Initial matrix\n");
    for( int i = 0; i < 4*4*SEVEN*SEVEN*SEVEN*SEVEN; i++ )
      I[i] = i;
    for( int i = 0; i < 4*SEVEN*SEVEN; i++ ) {
      for( int j = 0; j < 4*SEVEN*SEVEN; j++ )
	printf("%.0f ", I[i+j*4*SEVEN*SEVEN]);
      printf("\n");
    }
  }
  double O[numEntriesPerProc(desc)*SEVEN];
  distributeFrom1Proc( desc, O, I );
  for( int p = 0; p < desc.nproc; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < numEntriesPerProc(desc); i++ )
	printf("%.0f ", O[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  for( int i = numEntriesPerProc(desc); i < numEntriesPerProc(desc)*SEVEN; i++ )
    O[i] = 0;
  // now call gathermatrices:
  
  MatDescriptor desc2 = desc;
  desc2.nproc /= SEVEN;
  desc2.nprocr /= SEVEN;
  
  double work[numEntriesPerProc(desc)*SEVEN];
  double O2[numEntriesPerProc(desc)*SEVEN];
  
  gatherMatrices( desc, O, desc2, O2, work );

  for( int p = 0; p < desc.nproc; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < numEntriesPerProc(desc2); i++ )
	printf("%.0f ", O2[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  */
  // create a 2*SEVENx2*SEVEN matrix and send distribute it with bs=2.  Assumes we run with SEVEN procs
  /*
  MatDescriptor desc;
  desc.lda = 2*SEVEN;
  desc.nrec = 0;
  desc.nproc = SEVEN;
  desc.nprocr = SEVEN;
  desc.nprocc = 1;
  desc.bs = 2;
  int rank = getRank(SEVEN);
  double I[4*SEVEN*SEVEN];
  if( rank == 0 ) {
    for( int i = 0; i < 4*SEVEN*SEVEN; i++ )
      I[i] = i;
    for( int i = 0; i < 2*SEVEN; i++ ) {
      for( int j = 0; j < 2*SEVEN; j++ )
	printf("%f ", I[i+j*2*SEVEN]);
      printf("\n");
    }
  }
  double O[4*SEVEN];
  distributeFrom1Proc( desc, O, I );
  for( int p = 0; p < SEVEN; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < 4*SEVEN; i++ )
	printf("%f ", O[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  // create a SEVENxSEVEN matrix and send distribute it.  Assumes we run with SEVEN procs
  MatDescriptor desc;
  desc.lda = SEVEN;
  desc.nrec = 0;
  desc.nproc = SEVEN;
  desc.nprocr = SEVEN;
  desc.nprocc = 1;
  desc.bs = 1;
  int rank = getRank(SEVEN);
  double I[SEVEN*SEVEN];
  if( rank == 0 ) {
    for( int i = 0; i < SEVEN*SEVEN; i++ )
      I[i] = i;
    for( int i = 0; i < SEVEN; i++ ) {
      for( int j = 0; j < SEVEN; j++ )
	printf("%f ", I[i+j*SEVEN]);
      printf("\n");
    }
  }
  double O[SEVEN];
  distributeFrom1Proc( desc, O, I );
  for( int p = 0; p < SEVEN; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < SEVEN; i++ )
	printf("%f ", O[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  */

  /*
  // create a 2*SEVENx2*SEVEN matrix and send distribute it.  Assumes we run with SEVEN procs
  MatDescriptor desc;
  desc.lda = 2*SEVEN;
  desc.nrec = 1;
  desc.nproc = SEVEN;
  desc.nprocr = 1;
  desc.nprocc = SEVEN;
  int rank = getRank(SEVEN);
  desc.bs = 1;
  double I[4*SEVEN*SEVEN];
  if( rank == 0 ) {
    for( int i = 0; i < 2*SEVEN*2*SEVEN; i++ )
      I[i] = i;
    for( int i = 0; i < 2*SEVEN; i++ ) {
      for( int j = 0; j < 2*SEVEN; j++ )
	printf("%f ", I[i+j*2*SEVEN]);
      printf("\n");
    }
  }
  double O[4*SEVEN];
  distributeFrom1Proc( desc, O, I );
  for( int p = 0; p < SEVEN; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < 4*SEVEN; i++ )
	printf("%f ", O[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  */
  
  // create seven SEVENxSEVEN matrices and send distribute them.  Assumes we run with SEVEN procs.  The first one has numbers 0..48 in column major order, the rest are all zeros.
  /*
  MatDescriptor desc;
  desc.lda = SEVEN;
  desc.nrec = 0;
  desc.nproc = SEVEN;
  desc.nprocr = SEVEN;
  desc.nprocc = 1;
  int rank = getRank(SEVEN);
  desc.bs = 1;
  double data[SEVEN*SEVEN*SEVEN];
  double *I = data;
  if( rank == 0 ) {
    for( int i = 0; i < SEVEN*SEVEN; i++ ) {
      I[i] = i;
    }
    for( int i = SEVEN*SEVEN; i < SEVEN*SEVEN*SEVEN; i++ )
      data[i] = 0;
    for( int i = 0; i < SEVEN; i++ ) {
      for( int j = 0; j < SEVEN; j++ )
	printf("%f ", I[i+j*SEVEN]);
      printf("\n");
    }
  }
  double OData[SEVEN*SEVEN];
  double *O = OData;
  if( rank == 0 )
    printf("Calling distribute\n");
  distributeFrom1Proc( desc, O, I );
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 )
    printf("distribute returned\n");

  for( int i = SEVEN; i < SEVEN*SEVEN; i++ )
    O[i] = 0;
  for( int p = 0; p < SEVEN; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < SEVEN; i++ )
	printf("%f ", O[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }

  // Now call gatherMatrices to put these matrices onto fewer computers
  // this describes them after re-arrangement
  MatDescriptor desc2 = desc;
  desc2.nprocr = 1;
  desc2.rank = 0;
  desc2.nproc = 1;
  double Work[SEVEN*SEVEN];
  double Output[SEVEN*SEVEN];
  for( int i = 0; i < SEVEN*SEVEN; i++ )
    Output[i] = 0;
  if( rank == 0 )
    printf("CALLING gatheMatrices\n");
  MPI_Barrier( MPI_COMM_WORLD );
  gatherMatrices( desc, OData, desc2, Output, Work );
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 ) {
    printf("Call returned\n");
  }
  MPI_Barrier( MPI_COMM_WORLD );
  for( int p = 0; p < SEVEN; p++ ) {
    if( rank == p ) {
      printf("2) rank %d: ", rank);
      for( int i = 0; i < SEVEN*SEVEN; i++ )
	printf("%f ", Output[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  */

  // create a a 2*SEVENx2*SEVEN matrices and send distribute it.  Assumes we run with SEVEN*SEVEN procs.  It has numbers 0..195 in column major order.  Also create 6 more distributed matrices of zeros
  /*
  MatDescriptor desc;
  desc.lda = 2*SEVEN;
  desc.nrec = 1;
  desc.bs = 1;
  // other good tests: nrec=1,bs=1 or nrec=0,bs=1
  desc.nproc = SEVEN*SEVEN;
  desc.nprocr = SEVEN;
  desc.nprocc = SEVEN;
  int rank = getRank(SEVEN*SEVEN);
  double I[4*SEVEN*SEVEN];
  if( rank == 0 ) {
    for( int i = 0; i < 2*SEVEN*2*SEVEN; i++ ) {
      I[i] = i;
    }
    for( int i = 0; i < 2*SEVEN; i++ ) {
      for( int j = 0; j < 2*SEVEN; j++ )
	printf("%f ", I[i+j*2*SEVEN]);
      printf("\n");
    }
  }
  double OData[4*SEVEN];
  double *O = OData;
  distributeFrom1Proc( desc, O, I );
  for( int p = 0; p < SEVEN*SEVEN; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < 4; i++ )
	printf("%f ", O[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }

  // Now call gatherMatrices to put the first seven blocks of these matrices onto fewer computers
  // this describes them after re-arrangement
  MatDescriptor desc3 = desc;
  desc3.nprocc = 1;
  desc3.nprocr = SEVEN;
  desc3.nproc = SEVEN;
  double Work[SEVEN*4];
  double Output[SEVEN*4];
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 )
    printf("CALLING gatheMatrices\n");
  MPI_Barrier( MPI_COMM_WORLD );
  gatherMatrices( desc, OData, desc3, Output, Work );
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 ) {
    printf("Call returned\n");
  }
  MPI_Barrier( MPI_COMM_WORLD );
  for( int p = 0; p < SEVEN*SEVEN; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < SEVEN*4; i++ )
	printf("%f ", Output[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }

  */
  // a 4*SEVENx4*SEVEN matrix, allowing us to set nrec=2 would be the next natural test
  /*
  MatDescriptor desc;
  desc.lda = 4*SEVEN;
  desc.nrec = 0;
  desc.bs = 1;
  desc.nproc = SEVEN*SEVEN;
  desc.nprocr = SEVEN;
  desc.nprocc = SEVEN;
  int rank = getRank(SEVEN*SEVEN);

  double I[16*SEVEN*SEVEN];
  if( rank == 0 ) {
    for( int i = 0; i < 4*SEVEN*4*SEVEN; i++ ) {
      I[i] = i;
    }
    for( int i = 0; i < 4*SEVEN; i++ ) {
      for( int j = 0; j < 4*SEVEN; j++ )
	printf("%f ", I[i+j*4*SEVEN]);
      printf("\n");
    }
  }
  //MPI_Barrier(MPI_COMM_WORLD);
  double OData[16*SEVEN];
  double *O = OData;
  if( rank == 0 )
    printf("Calling distributFrom1Proc\n");
  distributeFrom1Proc( desc, O, I );
  if( rank == 0 )
    printf("returned from distributFrom1Proc\n");
  for( int p = 0; p < SEVEN*SEVEN; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < 16; i++ )
	printf("%f ", O[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  // Now call gatherMatrices to put the first seven blocks of these matrices onto fewer computers
  // this describes them after re-arrangement
  MatDescriptor desc3 = desc;
  desc3.nprocc = 1;
  desc3.nprocr = SEVEN;
  desc3.nproc = SEVEN;
  double Work[SEVEN*16];
  double Output[SEVEN*16];
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 )
    printf("CALLING gatheMatrices\n");
  MPI_Barrier( MPI_COMM_WORLD );
  gatherMatrices( desc, OData, desc3, Output, Work );
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 ) {
    printf("Call returned\n");
  }
  MPI_Barrier( MPI_COMM_WORLD );
  for( int p = 0; p < SEVEN*SEVEN; p++ ) {
    if( rank == p ) {
      printf("2) rank %d: ", rank);
      for( int i = 0; i < SEVEN*16; i++ )
	printf("%f ", Output[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }
  if( rank == 0 )
    printf("CALLING scatterMatrices\n");
  MPI_Barrier( MPI_COMM_WORLD );
  scatterMatrices( desc3, Output, desc, OData, Work );
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 ) {
    printf("Call returned\n");
  }
  MPI_Barrier( MPI_COMM_WORLD );
  for( int p = 0; p < SEVEN*SEVEN; p++ ) {
    if( rank == p ) {
      printf("3) rank %d: ", rank);
      for( int i = 0; i < 16; i++ )
	printf("%f ", OData[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }

  memset( I, 0., 16*SEVEN*SEVEN*sizeof(double));
  if( rank == 0 )
    printf("Calling collectTo1Proc\n");
  MPI_Barrier(MPI_COMM_WORLD);
  printf("Before: rank %d\n", rank);
  collectTo1Proc( desc, I, O );
  MPI_Barrier(MPI_COMM_WORLD);
  printf("After: rank %d\n", rank);
  if( rank == 0 )
    printf("returned from collectTo1Proc\n");
  if( rank == 0 ) {
    for( int i = 0; i < 4*SEVEN*4*SEVEN; i++ ) {
      I[i] = i;
    }
    for( int i = 0; i < 4*SEVEN; i++ ) {
      for( int j = 0; j < 4*SEVEN; j++ )
	printf("%f ", I[i+j*4*SEVEN]);
      printf("\n");
    }
  }
  */
  
  // next natural test: increase the number of processors involved beyond SEVEN*SEVEN
  // for use with SEVEN^3 processors
  /*
  MatDescriptor desc;
  desc.lda = SEVEN*SEVEN;
  desc.nrec = 0;
  desc.bs = 1;
  desc.nproc = SEVEN*SEVEN*SEVEN;
  desc.nprocr = SEVEN*SEVEN;
  desc.nprocc = SEVEN;
  int rank = getRank(SEVEN*SEVEN*SEVEN);

  double I[SEVEN*SEVEN*SEVEN*SEVEN];
  if( rank == 0 ) {
    for( int i = 0; i < SEVEN*SEVEN*SEVEN*SEVEN; i++ ) {
      I[i] = i;
    }
    for( int i = 0; i < SEVEN*SEVEN; i++ ) {
      for( int j = 0; j < SEVEN*SEVEN; j++ )
	printf("%f ", I[i+j*SEVEN*SEVEN]);
      printf("\n");
    }
  }
  //MPI_Barrier(MPI_COMM_WORLD);
  double OData[SEVEN*SEVEN];
  double *O = OData;
  if( rank == 0 )
    printf("Calling distributFrom1Proc\n");
  distributeFrom1Proc( desc, O, I );
  if( rank == 0 )
    printf("returned from distributFrom1Proc\n");
  for( int p = 0; p < SEVEN*SEVEN*SEVEN; p++ ) {
    if( rank == p ) {
      printf("rank %d: ", rank);
      for( int i = 0; i < SEVEN; i++ )
	printf("%f ", O[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }

  // Now call gatherMatrices to put the first seven blocks of these matrices onto fewer computers
  // this describes them after re-arrangement
  MatDescriptor desc3 = desc;
  desc3.nprocc = SEVEN;
  desc3.nprocr = SEVEN;
  desc3.nproc = SEVEN*SEVEN;
  double Work[SEVEN*SEVEN];
  double Output[SEVEN*SEVEN];
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 )
    printf("CALLING gatheMatrices\n");
  MPI_Barrier( MPI_COMM_WORLD );
  gatherMatrices( desc, OData, desc3, Output, Work );
  MPI_Barrier( MPI_COMM_WORLD );
  if( rank == 0 ) {
    printf("Call returned\n");
  }
  MPI_Barrier( MPI_COMM_WORLD );
  for( int p = 0; p < SEVEN*SEVEN; p++ ) {
    if( rank == p ) {
      printf("2) rank %d: ", rank);
      for( int i = 0; i < SEVEN*SEVEN; i++ )
	printf("%f ", Output[i] );
      printf("\n");
    }
    MPI_Barrier( MPI_COMM_WORLD );
  }

  */

  MPI_Finalize();
}
