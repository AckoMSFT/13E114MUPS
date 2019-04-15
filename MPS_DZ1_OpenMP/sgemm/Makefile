# C compiler
CC = g++-8
CC_FLAGS = -fopenmp -O2

prime: 
	$(CC) $(CC_FLAGS) sgemm.cc -o sgemm -lm 

clean:
	rm -f sgemm result*
