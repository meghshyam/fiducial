CC=g++
CFLAGS=`pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`

all:detectFiducial
	
detectFiducial: main.o GaborFilter.o
	$(CC) -o detectFiducial main.o GaborFilter.o $(LDFLAGS) -lm
	
main.o: src/main.cpp
	$(CC) -c -g -o main.o src/main.cpp $(CCFLAGS) 
	 
GaborFilter.o: src/GaborFilter.cpp
	$(CC) -c -g -o GaborFilter.o src/GaborFilter.cpp $(CCFLAGS)
	
clean:
	rm -rf *.o detectFiducial
	
	
 