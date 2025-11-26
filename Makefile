CC=gcc
CFLAGS=-O2 -Wall -std=c11

OBJS=main.o btree.o hctree.o

all: hctree_demo

hctree_demo: $(OBJS)
	$(CC) $(CFLAGS) -o hctree_demo $(OBJS) -lm

main.o: main.c btree.h hctree.h
btree.o: btree.c btree.h
hctree.o: hctree.c hctree.h btree.h

clean:
	rm -f $(OBJS) hctree_demo
