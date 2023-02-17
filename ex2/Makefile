CC = gcc
CFLAGS = -Wall -Werror

EXEC = main
SRC = $(wildcard *.c)
DAT = $(wildcard *.dat)

all: ${EXEC} clean

${EXEC}: ${SRC}
	${CC} ${CFLAGS} ${SRC} -o ${EXEC}

run: all
	./${EXEC}
	gnuplot plotting.gpi

clean:
	rm -f ${DAT}
