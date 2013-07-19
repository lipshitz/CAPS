CC=mpic++ -std=c++0x
CFLAGS = -O3

TESTS=test-generate test-multiply test-outer test-row test-improw test-summa test-iterative test-recursive test-lgen
BENCH=bench-multiply bench-outer bench-row bench-improw bench-summa bench-iterative bench-recursive
TARGETS=$(TESTS) $(BENCH)

all: $(TARGETS)

tests:	$(TESTS)

test-generate:	test-generate.cpp generate.o generate.h types.h
	$(CC) $(CFLAGS) $(LIBS) test-generate.cpp -o test-generate generate.o

test-multiply:	test-multiply.cpp generate.o generate.h types.h local-multiply.o library.o
	$(CC) $(CFLAGS) $(LIBS) test-multiply.cpp -o test-multiply generate.o local-multiply.o library.o

test-outer:	test-outer.cpp generate.o generate.h types.h local-multiply.o outer.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) test-outer.cpp -o test-outer generate.o local-multiply.o outer.o testlib.o library.o merge.o

test-row:	test-row.cpp generate.o generate.h types.h local-multiply.o row.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) test-row.cpp -o test-row generate.o local-multiply.o row.o testlib.o library.o merge.o

test-improw:	test-row.cpp generate.o generate.h types.h local-multiply.o improw.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) test-row.cpp -o test-improw generate.o local-multiply.o improw.o testlib.o library.o merge.o

test-summa:	test-summa.cpp generate.o generate.h types.h local-multiply.o summa.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) test-summa.cpp -o test-summa generate.o local-multiply.o summa.o testlib.o library.o merge.o

test-iterative:	test-iterative.cpp generate.o generate.h types.h local-multiply.o iterative.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) test-iterative.cpp -o test-iterative generate.o local-multiply.o iterative.o testlib.o library.o merge.o

test-recursive:	test-recursive.cpp generate.o generate.h types.h local-multiply.o recursive.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) test-recursive.cpp -o test-recursive generate.o local-multiply.o recursive.o testlib.o library.o merge.o

test-lgen:	test-lgen.cpp generate.o generate.h types.h local-multiply.o row.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) test-lgen.cpp -o test-lgen generate.o local-multiply.o row.o testlib.o library.o merge.o

bench-multiply:	bench-multiply.cpp generate.o generate.h types.h local-multiply.o library.o
	$(CC) $(CFLAGS) $(LIBS) bench-multiply.cpp -o bench-multiply generate.o local-multiply.o library.o

bench-outer:	bench-outer.cpp generate.o generate.h types.h local-multiply.o outer.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) bench-outer.cpp -o bench-outer generate.o local-multiply.o outer.o testlib.o library.o merge.o

bench-row:	bench-row.cpp generate.o generate.h types.h local-multiply.o row.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) bench-row.cpp -o bench-row generate.o local-multiply.o row.o testlib.o library.o merge.o

bench-improw:	bench-row.cpp generate.o generate.h types.h local-multiply.o improw.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) bench-row.cpp -o bench-improw generate.o local-multiply.o improw.o testlib.o library.o merge.o

bench-summa:	bench-summa.cpp generate.o generate.h types.h local-multiply.o summa.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) bench-summa.cpp -o bench-summa generate.o local-multiply.o summa.o testlib.o library.o merge.o

bench-iterative:	bench-iterative.cpp generate.o generate.h types.h local-multiply.o iterative.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) bench-iterative.cpp -o bench-iterative generate.o local-multiply.o iterative.o testlib.o library.o merge.o

bench-recursive:	bench-recursive.cpp generate.o generate.h types.h local-multiply.o recursive.o testlib.o library.o merge.o
	$(CC) $(CFLAGS) $(LIBS) bench-recursive.cpp -o bench-recursive generate.o local-multiply.o recursive.o testlib.o library.o merge.o

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm $(TARGETS) *.o

