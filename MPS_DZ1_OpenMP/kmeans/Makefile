# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O0 

kmeans: openmp_cluster.o cluster.o getopt.o kmeans.o openmp_kmeans_clustering.o kmeans_clustering.o 
	$(CC) $(CC_FLAGS) openmp_cluster.o cluster.o getopt.o kmeans.o openmp_kmeans_clustering.o kmeans_clustering.o  -o kmeans

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

openmp_cluster.o: openmp_cluster.c
	$(CC) $(CC_FLAGS) openmp_cluster.c -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c
	
getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c
	
kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) kmeans.c -c

openmp_kmeans_clustering.o: openmp_kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) openmp_kmeans_clustering.c -c

kmeans_clustering.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c

clean:
	rm -f *.o *~ kmeans 
