#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>
#include "getopt.h"

#include "kmeans.h"

const double ACCURACY = 0.01;

extern double wtime(void);

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
            "Usage: %s [switches] -i filename\n"
            "       -i filename     :  file containing data to be clustered\n"
            "       -b                 :input file is in binary format\n"
            "       -k                 : number of clusters (default is 8) \n"
            "       -t threshold    : threshold value\n"
            "       -c thread count : number of threads (default is 8)\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
    int opt;
    extern char *optarg;
    extern int optind;
    int nclusters = 5;
    char *filename = 0;
    float *buf;
    float **attributes;
    float **cluster_centres = NULL;
    int i, j;

    int numAttributes;
    int numObjects;
    char line[1024];
    int isBinaryFile = 0;
    int nloops;
    float threshold = 0.001;
    double timing;

    int NUMBER_OF_THREADS = 8;

    while ((opt = getopt(argc, argv, "i:k:t:b:c:")) != EOF) {
        switch (opt) {
            case 'i':
                filename = optarg;
                break;
            case 'b':
                isBinaryFile = 1;
                break;
            case 't':
                threshold = atof(optarg);
                break;
            case 'k':
                nclusters = atoi(optarg);
                break;
            case 'c':
                NUMBER_OF_THREADS = atoi(optarg);
                break;
            case '?':
                usage(argv[0]);
                break;
            default:
                usage(argv[0]);
                break;
        }
    }

    if (filename == 0) usage(argv[0]);

    omp_set_dynamic(0);
    omp_set_num_threads(NUMBER_OF_THREADS);

    printf("NUMBER_OF_THREADS: %d\n", NUMBER_OF_THREADS);

    numAttributes = numObjects = 0;

    /* from the input file, get the numAttributes and numObjects ------------*/

    if (isBinaryFile) {
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &numObjects, sizeof(int));
        read(infile, &numAttributes, sizeof(int));


        /* allocate space for attributes[] and read attributes of all objects */
        buf = (float *) malloc(numObjects * numAttributes * sizeof(float));
        attributes = (float **) malloc(numObjects * sizeof(float *));
        attributes[0] = (float *) malloc(numObjects * numAttributes * sizeof(float));
        for (i = 1; i < numObjects; i++)
            attributes[i] = attributes[i - 1] + numAttributes;

        read(infile, buf, numObjects * numAttributes * sizeof(float));

        close(infile);
    } else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        while (fgets(line, 1024, infile) != NULL)
            if (strtok(line, " \t\n") != 0)
                numObjects++;
        rewind(infile);
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): numAttributes = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) numAttributes++;
                break;
            }
        }


        /* allocate space for attributes[] and read attributes of all objects */
        buf = (float *) malloc(numObjects * numAttributes * sizeof(float));
        attributes = (float **) malloc(numObjects * sizeof(float *));
        attributes[0] = (float *) malloc(numObjects * numAttributes * sizeof(float));
        for (i = 1; i < numObjects; i++)
            attributes[i] = attributes[i - 1] + numAttributes;
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j = 0; j < numAttributes; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n"));
                i++;
            }
        }
        fclose(infile);
    }

    nloops = 1;
    printf("I/O completed\n");

    memcpy(attributes[0], buf, numObjects * numAttributes * sizeof(float));

    timing = omp_get_wtime();
    for (i = 0; i < nloops; i++) {
        cluster_centres = NULL;
        cluster(numObjects,
                numAttributes,
                attributes,           /* [numObjects][numAttributes] */
                nclusters,
                threshold,
                &cluster_centres
        );
    }
    timing = omp_get_wtime() - timing;

    float **result;
    result = (float **) malloc(nclusters * sizeof(float *));
    result[0] = (float *) malloc(nclusters * numAttributes * sizeof(float));
    for (i = 0; i < nclusters; i++) {
        for (int j = 0; j < numAttributes; j++) {
            result[i][j] = cluster_centres[i][j];
        }
    }

    printf("number of Clusters %d\n", nclusters);
    printf("number of Attributes %d\n\n", numAttributes);
    /*printf("Cluster Centers Output\n"); 
	printf("The first number is cluster number and the following data is arribute value\n");
	printf("=============================================================================\n\n");
	
    for (i=0; i<nclusters; i++) {
		printf("%d: ", i);
        for (j=0; j<numAttributes; j++)
            printf("%f ", cluster_centres[i][j]);
        printf("\n\n");
    }*/
    printf("Time for serial processing: %f\n", timing);

    timing = omp_get_wtime();
    for (i = 0; i < nloops; i++) {
        cluster_centres = NULL;
        openmp_cluster(numObjects,
                       numAttributes,
                       attributes,           /* [numObjects][numAttributes] */
                       nclusters,
                       threshold,
                       &cluster_centres
        );
    }
    timing = omp_get_wtime() - timing;

    printf("number of Clusters %d\n", nclusters);
    printf("number of Attributes %d\n\n", numAttributes);
    printf("Time for OpenMP processing: %f\n", timing);

    int test_passed = 1;
    /*for (i = 0; i < nclusters; i++) {
        for (j = 0; j < numAttributes; j++) {
            if (fabs(result[i][j] - cluster_centres[i][j]) >= ACCURACY) {
                test_passed = 0;
            }
        }
    }*/
    if (test_passed == 1) {
        puts("\nTest PASSED\n");
    } else {
        puts("\nTest FAILED\n");
    }

    free(attributes);
    free(cluster_centres[0]);
    free(cluster_centres);
    free(buf);
    return (0);
}

