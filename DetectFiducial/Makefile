CC=g++
CFLAGS=`pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`

all:detectFiducial
	
detectFiducial: main.o GaborFilter.o ConnectedComponents.o
	$(CC) -o detectFiducial main.o GaborFilter.o ConnectedComponents.o $(LDFLAGS) -lm -L"/home/meghshyam/workspace/AlgLibShared/Debug" -lAlgLibShared
	
main.o: src/main.cpp src/ConnectedComponents.h src/GaborFilter.h
	$(CC) -c -g -o main.o src/main.cpp $(CCFLAGS) -I"/home/meghshyam/workspace/AlgLibShared"
	 
GaborFilter.o: src/GaborFilter.cpp src/GaborFilter.h
	$(CC) -c -g -o GaborFilter.o src/GaborFilter.cpp $(CCFLAGS)

ConnectedComponents.o: src/ConnectedComponents.cpp src/ConnectedComponents.h
	$(CC) -c -g -o ConnectedComponents.o src/ConnectedComponents.cpp $(CCFLAGS)


clean:
	rm -rf *.o detectFiducial
	
	
 