#include "counters.h"
#include "communication.h"
//#include "memory.h"
#include "library.h"

long long messages = 0;
long long words = 0;
double timers[NUM_TIMERS];

void initTimers() {
  for( int i = 0; i < NUM_TIMERS; i++ )
    timers[i] = 0;
  messages = 0;
  words = 0;
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

void increaseMessages(long long n) {
  messages += n;
}

void increaseWords(long long n) {
  words += n;
}

void printCounters(int n) {
  printf("Rank %d time on comm %f+%f+%f+%f, re-arrange %f+%f+%f+%f, base %f+%f+%f+%f\n", getRank(), timers[0], timers[1], timers[2], timers[3], timers[4], timers[5], timers[6], timers[7], timers[8], timers[9], timers[10], timers[11]);
}
