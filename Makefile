main: layer.o util.o mnist_read.o network.o main.c main.h
	cc -lm layer.o util.o mnist_read.o network.o main.c

layer.o: layer.c
	cc -c layer.c

util.o: util.c
	cc -lm -c util.c

mnist_read.o: mnist_read.c
	cc -c mnist_read.c

network.o: network.c util.o main.h
	cc -c network.c

clean:
	rm *.o
