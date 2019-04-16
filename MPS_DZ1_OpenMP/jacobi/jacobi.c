#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const double ACCURACY = 0.01;
int NUMBER_OF_THREADS = 8;

double* serial_jacobi_implementation(int m, int n) {
    double *b;
    double d;
    int i;
    int it;
    double r;
    double t;
    double *x;
    double *xnew;

    b = (double *) malloc(n * sizeof(double));
    x = (double *) malloc(n * sizeof(double));
    xnew = (double *) malloc(n * sizeof(double));

    for (i = 0; i < n; i++) {
        b[i] = 0.0;
    }

    b[n - 1] = (double) (n + 1);
    for (i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    for (it = 0; it < m; it++) {
        for (i = 0; i < n; i++) {
            xnew[i] = b[i];
            if (0 < i) {
                xnew[i] = xnew[i] + x[i - 1];
            }
            if (i < n - 1) {
                xnew[i] = xnew[i] + x[i + 1];
            }
            xnew[i] = xnew[i] / 2.0;
        }
        d = 0.0;
        for (i = 0; i < n; i++) {
            d = d + pow(x[i] - xnew[i], 2);
        }
        for (i = 0; i < n; i++) {
            x[i] = xnew[i];
        }
        r = 0.0;
        for (i = 0; i < n; i++) {
            t = b[i] - 2.0 * x[i];
            if (0 < i) {
                t = t + x[i - 1];
            }
            if (i < n - 1) {
                t = t + x[i + 1];
            }
            r = r + t * t;
        }
    }
    free(b);
    free(xnew);

    return x;
}

double* openmp_jacobi_implementation(int m, int n) {
    double *b;
    double d;
    int i;
    int it;
    double r;
    double t;
    double *x;
    double *xnew;

    b = (double *) malloc(n * sizeof(double));
    x = (double *) malloc(n * sizeof(double));
    xnew = (double *) malloc(n * sizeof(double));

#pragma omp parallel private(i)
    {
#pragma omp for
        for (i = 0; i < n; i++) {
            b[i] = 0.0;
        }
#pragma omp single
        b[n - 1] = (double) (n + 1);
#pragma omp for
        for (i = 0; i < n; i++) {
            x[i] = 0.0;
        }
    }

    for (it = 0; it < m; it++) {
#pragma omp parallel private(i, t)
        {
#pragma omp for
            for (i = 0; i < n; i++) {
                xnew[i] = b[i];
                if (0 < i) {
                    xnew[i] = xnew[i] + x[i - 1];
                }
                if (i < n - 1) {
                    xnew[i] = xnew[i] + x[i + 1];
                }
                xnew[i] = xnew[i] / 2.0;
            }

            d = 0.0;
#pragma omp for reduction (+ : d)
            for (i = 0; i < n; i++) {
                d = d + pow(x[i] - xnew[i], 2);
            }

#pragma omp for
            for (i = 0; i < n; i++) {
                x[i] = xnew[i];
            }

            r = 0.0;
#pragma omp for reduction (+ : r)
            for (i = 0; i < n; i++) {
                t = b[i] - 2.0 * x[i];
                if (0 < i) {
                    t = t + x[i - 1];
                }
                if (i < n - 1) {
                    t = t + x[i + 1];
                }
                r = r + t * t;
            }
        }
    }

    free(b);
    free(xnew);

    return x;
}

int main(int argc, char *argv[]) {
    int m, n;
    if (argc == 3 || argc == 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    } else {
        m = 5000;
        n = 50000;
    }
    if (argc == 4) {
        NUMBER_OF_THREADS = atoi(argv[3]);
    }

    omp_set_dynamic(false);
    omp_set_num_threads(NUMBER_OF_THREADS);

    printf("NUMBER_OF_THREADS: %d\n", NUMBER_OF_THREADS);

    //printf("\n");
    //printf("JACOBI_OPENMP:\n");
    //printf("  C/OpenMP version\n");
    //printf("  Jacobi iteration to solve A*x=b.\n");
    //printf("\n");
    printf("  Number of variables  N = %d\n", n);
    printf("  Number of iterations M = %d\n", m);
    //printf("\n");

    double wall_clock = omp_get_wtime(), time_elapsed;

    // serial implementation
    double *result_serial = serial_jacobi_implementation(m, n);
    time_elapsed = omp_get_wtime() - wall_clock;
    printf("serial_implementation: time_elapsed: %lf seconds\n", time_elapsed);


    // OpenMP implementation
    wall_clock = omp_get_wtime();
    double *result_parallel = openmp_jacobi_implementation(m, n);
    time_elapsed = omp_get_wtime() - wall_clock;
    //printf("\n");

    printf("parallel_implementation: time_elapsed: %lf seconds\n", time_elapsed);

    //printf("\n");

    bool equal = true;
    for (int i = 0; i < n; i++) {
        if (fabs(result_serial[i] - result_parallel[i]) >= ACCURACY) {
            equal = false;
        }
    }
    if (equal) {
        puts("Test PASSED");
    } else {
        puts("Test FAILED");
    }

    printf("\n");
    return 0;
}

