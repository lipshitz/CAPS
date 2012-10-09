#ifndef COUNTERS_H
#define COUNTERS_H
#include "matrix.h"

void increaseAdditions(long long n);
void increaseMultiplications(long long n);
void increaseMessages(long long n);
void increaseWords(long long n);
void setExecutionType(int nrec, const char* type);
void initTimers();
void startTimer(int type);
void stopTimer(int type);
const int TIMER_COMM = 0;
const int TIMER_ADD = 1;
const int TIMER_MUL = 2;
const int TIMER_REARRANGE = 3;
const int TIMER_COMM_SUMMA = 4;
const int NUM_TIMERS = 5;

void printCounters(double t, MatDescriptor desc);
void printCounters(MatDescriptor desc);
#endif
