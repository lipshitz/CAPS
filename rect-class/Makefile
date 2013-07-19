CC=mpic++
CFLAGS=-O3 -ffast-math -funroll-loops -fopenmp
LIBS=-lblas -llapack

TARGETS=bench-rect bench-rect-nc

all:	$(TARGETS)

bench-rect:	bench-rect.cpp communication.o library.o rectsizes.o rectmm.o blas.h communication.h rectmm.h library.h rectsizes.h sizes.o
	$(CC) $(CFLAGS) bench-rect.cpp -o bench-rect communication.o library.o rectsizes.o rectmm.o sizes.o $(LIBS)

bench-rect-nc:	bench-rect.cpp communication.o library.o rectsizes.o rectmm-nc.o blas.h communication.h rectmm.h library.h rectsizes.h sizes.o
	$(CC) $(CFLAGS) bench-rect.cpp -o bench-rect-nc communication.o library.o rectsizes.o rectmm-nc.o sizes.o $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

sequential:	sequential.cpp
	$(CC) sequential.cpp -o sequential $(LIBS)

clean:
	rm $(TARGETS) *.o
