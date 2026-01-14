CC=gcc
CFLAGS=-O3 -Wall -Wextra -std=c11 -pthread

all: test bench

test: test.o matmul.o
	$(CC) $(CFLAGS) -o test test.o matmul.o

bench: bench.o matmul.o
	$(CC) $(CFLAGS) -o bench bench.o matmul.o

test.o: test.c matmul.h
	$(CC) $(CFLAGS) -c test.c

bench.o: bench.c matmul.h
	$(CC) $(CFLAGS) -c bench.c

matmul.o: matmul.c matmul.h
	$(CC) $(CFLAGS) -c matmul.c

clean:
	rm -f *.o test bench
