CC	= nvcc
CPP	= nvcc
EXE	= mri-q
OBJ	= file.o parboil.o args.o main.o
#APP_CXXFLAGS=-O3 -funroll-all-loops -ffast-math
#APP_CFLAGS=-O3 -funroll-all-loops -ffast-math
APP_LDFLAGS=-lm -lstdc++
LANGUAGE=c

default: $(EXE)

main.o: main.cu computeQ.cu parboil.h
	$(CC) $(APP_CFLAGS) -c main.cu

parboil.o: parboil.c parboil.h
	$(CC) $(APP_CFLAGS) -c parboil.c

file.o: file.cc file.h
	$(CPP) $(APP_CXXFLAGS) -c file.cc

args.o: args.c 
	$(CC) $(APP_CFLAGS) -c args.c

mri-q: main.o file.o args.o parboil.o
	$(CPP) $(APP_CXXFLAGS) $(OBJ) -o $(EXE) $(APP_LDFLAGS)

clean:
	rm -f *.o $(EXE) 
