CC=gcc
CFLAGS=-g -fopenmp -Wall -Wpedantic
#CFLAGS=-fopenmp -Wall -Wpedantic -O3 -msse4.2

lab2: lab2.o wctimer.o
	$(CC) -fopenmp -o $@ $^ -lm

clean:
	rm -f *.o lab2
