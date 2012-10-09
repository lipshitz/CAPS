#include "counters.h"
#include "communication.h"
#include "memory.h"
#include "library.h"

long long additions = 0;
long long multiplications = 0;
long long messages = 0;
long long words = 0;
char* executionPattern;
double timers[NUM_TIMERS];
double startTime;

void initTimers() {
  for( int i = 0; i < NUM_TIMERS; i++ )
    timers[i] = 0;
}

void startTimer(int type) {
#ifdef SANITY_CHECKS
  assert(type < NUM_TIMERS);
#endif
  timers[type] -= read_timer();
}

void stopTimer(int type) {
#ifdef SANITY_CHECKS
  assert(type < NUM_TIMERS);
#endif
  timers[type] += read_timer();
}

void increaseAdditions(long long n) {
  additions += n;
}

void increaseMultiplications(long long n) {
  multiplications += n;
}

void increaseMessages(long long n) {
  messages += n;
}

void increaseWords(long long n) {
  words += n;
}

void setExecutionType(int nrec, const char* type) {
  static int nrecMax = -1;
  if( nrecMax == -1 ) {
    nrecMax = nrec;
    executionPattern = new char[4*(nrecMax)+1];
    executionPattern[4*(nrecMax)] = '\0';
  }
  for( int i = 0; i < 3; i++ )
    executionPattern[(nrecMax-nrec)*4+i] = type[i];
  executionPattern[(nrecMax-nrec)*4+3] = ',';
}

void printCounters(double t, MatDescriptor desc) {
  // we could print this for all processors, but it will be the same
  if( getRank() == 0 ) {
    int memoryUsed = getMaxUsedMemory();
    printf("Rank %d: peak memory usage %.2f MB\n", getRank(), memoryUsed*8./1024/1024);
    printf("Execution pattern %s\n", executionPattern);
    long long flops = additions+multiplications;
    printf("Rank %d: flops=%lld (%lld+%lld) messages sent+received=%lld bytes sent+received=%lld\n", getRank(), flops, additions-multiplications, 2*multiplications, messages, words); 
    printf("Rankd %d: time %f actual Gflop/s %f per process effective Gflop/s %f per process\n", getRank(), t, flops/1.e9/t, 2.*desc.lda*desc.lda*desc.lda/desc.nproc/desc.nproc_summa/1.e9/t);
  }
  MPI_Barrier( MPI_COMM_WORLD );
  // this we print for all processors
  printf("rank %d Time spent communicating: %f (%f+%f), on additions: %f, on multiplications %f, on local reording %f (sum %f)\n", getRank(), timers[TIMER_COMM]+timers[TIMER_COMM_SUMMA], timers[TIMER_COMM], timers[TIMER_COMM_SUMMA], timers[TIMER_ADD], timers[TIMER_MUL], timers[TIMER_REARRANGE], timers[TIMER_COMM] + timers[TIMER_ADD] + timers[TIMER_MUL] + timers[TIMER_REARRANGE] );
}

void printCounters(MatDescriptor desc) {
  // we could print this for all processors, but it will be the same
  double memoryUsed = getMaxUsedMemory();
  printf("peak memory usage %.2f MB\n", memoryUsed*8./1024/1024);
  printf("Execution pattern %s\n", executionPattern);
  long long flops = additions+multiplications;
  printf("flops=%lld messages sent+received=%lld bytes sent+received=%lld\n", flops, messages, words); 
}
