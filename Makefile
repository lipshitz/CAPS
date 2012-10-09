include make.inc

TESTS=mpitest commtest summatest
TARGETS=fromfile1 randombenchmark


all: $(TARGETS)

tests:	$(TESTS)

mpitest:	mpitest.cpp
	$(CC) mpitest.cpp -o mpitest $(BFLAGS)

summatest:	summatest.cpp summa1d.o summa1d.h matrix.o dgemm-blas.o dgemm-blas.h matrix.h communication.o communication.h
	$(CC) summatest.cpp matrix.o dgemm-blas.o communication.o counters.o memory.o library.o summa1d.o -o summatest $(BFLAGS)

commtest:	comm-test.cpp communication.h communication.o matrix.h matrix.o counters.o memory.o library.o
	$(CC) comm-test.cpp communication.o matrix.o counters.o memory.o library.o -o commtest $(BFLAGS)

fromfile1:	fromfile1.cpp communication.h communication.o matrix.h matrix.o library.h memory.h memory.o multiply.h multiply.o tags.h dgemm-blas.h dgemm-blas.o counters.h counters.o library.o summa1d.h summa1d.o
	$(CC) fromfile1.cpp communication.o matrix.o multiply.o memory.o -o fromfile1 dgemm-blas.o counters.o library.o summa1d.o $(BFLAGS)

randombenchmark:	randombenchmark.cpp communication.h communication.o matrix.h matrix.o library.h memory.h memory.o multiply.h multiply.o tags.h dgemm-blas.h dgemm-blas.o counters.h counters.o library.o summa1d.h summa1d.o
	$(CC) randombenchmark.cpp communication.o matrix.o multiply.o memory.o -o randombenchmark dgemm-blas.o counters.o library.o summa1d.o $(BFLAGS)

%.o: %.cpp
	$(CC) -c $(OFLAGS) $<

clean:
	rm -f $(TARGETS) $(TESTS) *.o
