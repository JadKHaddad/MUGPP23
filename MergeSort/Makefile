# debian version 12.0
# gcc (GCC) 13.1.0

CXX = g++
STD = -std=c++17
CFLAGS = -Wall -Wextra -pedantic -Wcast-align -Wpointer-arith -Wcast-qual -Wno-missing-braces -Werror -Wformat -Wformat-security

main: main.o
	$(CXX) $(STD) $(CFLAGS) -o main main.o

main.o: main.cpp
	$(CXX) $(STD) $(CFLAGS) -c main.cpp

release: CFLAGS += -O3 -static
release: main

clean:
	$(RM) *.o main

remake: clean main

remake_release: clean release