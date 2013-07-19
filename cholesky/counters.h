#ifndef COUNTERS_H
#define COUNTERS_H

//void increaseAdditions(long long n);
//void increaseMultiplications(long long n);
void increaseMessages(long long n);
void increaseWords(long long n);
//void setExecutionType(int nrec, const char* type);
void initTimers();
void startTimer(int type);
void stopTimer(int type);
const int TIMER_COMM_CHOL = 0;
const int TIMER_COMM_TRSM = 1;
const int TIMER_COMM_SYRK = 2;
const int TIMER_COMM_MULT = 3;
const int TIMER_REARRANGE_CHOL = 4;
const int TIMER_REARRANGE_TRSM = 5;
const int TIMER_REARRANGE_SYRK = 6;
const int TIMER_REARRANGE_MULT = 7;
const int TIMER_BASE_CHOL = 8;
const int TIMER_BASE_TRSM = 9;
const int TIMER_BASE_SYRK = 10;
const int TIMER_BASE_MULT = 11;
const int NUM_TIMERS = 12;

void printCounters(int n);
#endif
