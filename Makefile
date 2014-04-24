CPP = g++
CFLAGS = -Wall -Wextra -Werror -O3 -std=c++11
INC = -I/usr/include/eigen3

all: ridge

ridge: src/ridge.cpp
	$(CPP) $(CFLAGS) $(INC) src/ridge.cpp ${LIBS} -o bin/ridge

clean:
	rm bin/ridge

test:
	bin/ridge data/test.data

