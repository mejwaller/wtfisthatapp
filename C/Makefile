CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/
LIBS=jpeg

jpgresizer: jpgresizer.o
	$(CC) jpgresizer.o -o jpgresizer $(LDFLAGS) -l$(LIBS)

jpgresizer.o:
	$(CC) $(CFLAGS) -c jpgresizer.c -o jpgresizer.o

clean:
	-rm jpgresizer *.o
