CC = gcc
CFLAGS = -Wall -Werror

EXEC = main
SRC = $(wildcard *.c)
DAT = $(wildcard *.dat)

all: ${EXEC} clean

${EXEC}: ${SRC} mnist_data
	${CC} ${CFLAGS} ${SRC} -o ${EXEC}

mnist_data:
	cd $@
	make

run: all
	./${EXEC}
	gnuplot plotting.gpi

clean:
	rm -f ${DAT}
