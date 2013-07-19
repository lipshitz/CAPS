CC=mpic++
CFLAGS=-O3 -ffast-math -funroll-loops -fopenmp
LIBS=-lblas -llapack

TARGETS=test-trsm test-mult test-syrk test-chol sequential bench-chol bench-mult

all:	$(TARGETS)

test-trsm:	test-trsm.cpp communication.o trsm.o mult.o library.o sizes.o blas.h communication.h trsm.h mult.h library.h sizes.h counters.h counters.o
	$(CC) $(CFLAGS) test-trsm.cpp -o test-trsm communication.o trsm.o mult.o library.o sizes.o counters.o $(LIBS)

test-chol:	test-chol.cpp communication.o chol.o mult.o library.o sizes.o trsm.o syrk.o blas.h communication.h chol.h mult.h library.h sizes.h counters.h counters.o
	$(CC) $(CFLAGS) test-chol.cpp -o test-chol communication.o chol.o mult.o library.o sizes.o trsm.o syrk.o counters.o $(LIBS)

bench-chol:	bench-chol.cpp communication.o chol.o mult.o library.o sizes.o trsm.o syrk.o blas.h communication.h chol.h mult.h library.h sizes.h counters.h counters.o
	$(CC) $(CFLAGS) bench-chol.cpp -o bench-chol communication.o chol.o mult.o library.o sizes.o trsm.o syrk.o counters.o $(LIBS)

test-syrk:	test-syrk.cpp communication.o syrk.o mult.o library.o sizes.o blas.h communication.h syrk.h mult.h library.h sizes.h counters.h counters.o
	$(CC) $(CFLAGS) test-syrk.cpp -o test-syrk communication.o syrk.o mult.o library.o sizes.o counters.o $(LIBS)

test-mult:	test-mult.cpp communication.o mult.o library.o sizes.o blas.h communication.h mult.h library.h sizes.h counters.h counters.o
	$(CC) $(CFLAGS) test-mult.cpp -o test-mult communication.o mult.o library.o sizes.o counters.o $(LIBS)

bench-mult:	bench-mult.cpp communication.o mult.o library.o sizes.o blas.h communication.h mult.h library.h sizes.h counters.h counters.o
	$(CC) $(CFLAGS) bench-mult.cpp -o bench-mult communication.o mult.o library.o sizes.o counters.o $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

sequential:	sequential.cpp
	$(CC) sequential.cpp -o sequential $(LIBS)

clean:
	rm $(TARGETS) *.o
