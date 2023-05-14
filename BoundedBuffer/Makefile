CXX = gcc
STD = -std=c99
CFLAGS = -Wall -Wextra -pedantic -Wcast-align -Wpointer-arith -Wcast-qual -Wno-missing-braces -Wformat -Wformat-security
LIBS = -pthread

main: main.o
	$(CXX) $(CFLAGS) $(STD) -o main main.o $(LIBS)

main.o: main.c
	$(CXX) $(CFLAGS) $(STD) -c main.c

clean:
	$(RM) *.o main

remake: clean main