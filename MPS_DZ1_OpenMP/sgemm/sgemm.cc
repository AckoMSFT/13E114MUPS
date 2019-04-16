#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <functional>

const double ACCURACY = 0.01;
int NUMBER_OF_THREADS = 8;

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v) {
    std::cerr << "Opening file:" << fn << std::endl;
    std::fstream f(fn, std::fstream::in);
    if (!f.good()) {
        return false;
    }

    // Read # of rows and cols
    f >> nr_row;
    f >> nr_col;

    float data;
    std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
    while (f.good()) {
        f >> data;
        v.push_back(data);
    }
    v.pop_back(); // remove the duplicated last element
    return true;
}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float> &v) {
    std::cerr << "Opening file:" << fn << " for write." << std::endl;
    std::fstream f(fn, std::fstream::out);
    if (!f.good()) {
        return false;
    }

    // Read # of rows and cols
    f << nr_row << " " << nr_col << " ";

    std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
    for (float i : v) {
        f << i << ' ';
    }
    f << "\n";

    return true;
}

/* 
 * Base C implementation of MM
 */

void
basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb,
           float beta, float *C, int ldc) {
    if ((transa != 'N') && (transa != 'n')) {
        std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
        return;
    }

    if ((transb != 'T') && (transb != 't')) {
        std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
        return;
    }

    for (int mm = 0; mm < m; ++mm) {
        for (int nn = 0; nn < n; ++nn) {
            float c = 0.0f;
            for (int i = 0; i < k; ++i) {
                float a = A[mm + i * lda];
                float b = B[nn + i * ldb];
                c += a * b;
            }
            C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
        }
    }
}

void
sgemmNoWorksharing(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B,
                   int ldb, float beta, float *C, int ldc) {
    if ((transa != 'N') && (transa != 'n')) {
        std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
        return;
    }

    if ((transb != 'T') && (transb != 't')) {
        std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
        return;
    }

    int element_count = m * n;
    int div = element_count / NUMBER_OF_THREADS;
    int mod = element_count % NUMBER_OF_THREADS;

    // m je matArow
    // n je matBCol
    // k je zajednicko matACol matBRow
    // matricaB je transponovana
    // rezultat je oblika m x n

    //printf("element_count: %d\n", element_count);
    int thread_id;
#pragma omp parallel private(thread_id) num_threads(NUMBER_OF_THREADS)
    {
        thread_id = omp_get_thread_num();
        int lower_bound = div * thread_id + mod, upper_bound = lower_bound + div;
        if (thread_id == 0) {
            lower_bound -= mod;
        }
        //printf("thread_id: %d lower_bound: %d upper_bound: %d: \n", thread_id, lower_bound, upper_bound);
        //printf("lda: %d, ldb: %d, ldc: %d\n", lda, ldb, ldc);
        for (int i = lower_bound; i < upper_bound; i++) {
            float element = 0.0f;
            int offset_b = i / ldc, offset_a = i % ldc;
            // C[i][j] = sum { A[i][k] * B[k][j] }
            // A je dimenzija m x k
            // B je dimenzija k x n
            for (int offset_c = 0; offset_c < k; offset_c++) {
                int idxA = offset_a + offset_c * lda;
                int idxB = offset_b + offset_c * ldb;
                element += A[idxA] * B[idxB];
            }
            C[i] = C[i] * beta + alpha * element;
        }
    }
}

void
sgemmWorksharing(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B,
                 int ldb, float beta, float *C, int ldc) {
    if ((transa != 'N') && (transa != 'n')) {
        std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
        return;
    }

    if ((transb != 'T') && (transb != 't')) {
        std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
        return;
    }

    int mm, nn, i;
    // posmatraj petlje mm i nn kao jednu veliku petlju i ravnomerno raspodeli posao
#pragma omp parallel for private(mm, nn, i) shared(A, B, C) collapse(2)
    for (mm = 0; mm < m; ++mm) {
        for (nn = 0; nn < n; ++nn) {
            float c = 0.0f;
            for (i = 0; i < k; ++i) {
                float a = A[mm + i * lda];
                float b = B[nn + i * ldb];
                c += a * b;
            }
            C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
        }
    }
}

void sgemmTasking(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B,
                  int ldb, float beta, float *C, int ldc) {
    if ((transa != 'N') && (transa != 'n')) {
        std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
        return;
    }

    if ((transb != 'T') && (transb != 't')) {
        std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
        return;
    }

    int element_count = m * n;
    int div = element_count / NUMBER_OF_THREADS;
    int mod = element_count % NUMBER_OF_THREADS;

    // m je matArow
    // n je matBCol
    // k je zajednicko matACol matBRow
    // matricaB je transponovana
    // rezultat je oblika m x n

    //printf("element_count: %d\n", element_count);
    int thread_id;
#pragma omp parallel private(thread_id) num_threads(NUMBER_OF_THREADS)
    {
        thread_id = omp_get_thread_num();
        int lower_bound = div * thread_id + mod, upper_bound = lower_bound + div;
        if (thread_id == 0) {
            lower_bound -= mod;
        }
        //printf("thread_id: %d lower_bound: %d upper_bound: %d: \n", thread_id, lower_bound, upper_bound);
        //printf("lda: %d, ldb: %d, ldc: %d\n", lda, ldb, ldc);
        for (int i = lower_bound; i < upper_bound; i++) {
#pragma omp task
            {
                float element = 0.0f;
                int offset_b = i / ldc, offset_a = i % ldc;
                // C[i][j] = sum { A[i][k] * B[k][j] }
                // A je dimenzija m x k
                // B je dimenzija k x n
                for (int offset_c = 0; offset_c < k; offset_c++) {
                    int idxA = offset_a + offset_c * lda;
                    int idxB = offset_b + offset_c * ldb;
                    element += A[idxA] * B[idxB];
                }
                C[i] = C[i] * beta + alpha * element;
            }
        }
    }
}

bool replace(std::string &str, const std::string &from, const std::string &to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos) {
        return false;
    }
    str.replace(start_pos, from.length(), to);
    return true;
}

std::function<bool(const float &, const float &)> comparator = [](const float &lhs, const float &rhs) {
    return abs(lhs - rhs) < ACCURACY;
};

bool validate_result(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), comparator);
}

int main(int argc, char *argv[]) {
    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT;

    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Expecting three filenames or four arguments\n");
        exit(-1);
    }

    /* Read in data */
    // load A
    readColMajorMatrixFile(argv[1], matArow, matAcol, matA);

    // load B^T
    readColMajorMatrixFile(argv[2], matBcol, matBrow, matBT);

    // allocate space for C
    std::vector<float> matC(matArow * matBcol);

    omp_set_dynamic(false);
    if (argc == 5) {
        NUMBER_OF_THREADS = atoi(argv[4]);
    }
    omp_set_num_threads(NUMBER_OF_THREADS);

    printf("matA: %s, matB: %s, matC: %s\n", argv[1], argv[2], argv[3]);
    printf("NUMBER_OF_THREADS: %d\n", NUMBER_OF_THREADS);

    double wall_clock = omp_get_wtime(), time_elapsed;
    // Use standard sgemm interface
    basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f,
               &matC.front(), matArow);
    time_elapsed = omp_get_wtime() - wall_clock;
    printf("serial_implementation: time_elapsed: %lf seconds\n", time_elapsed);
    writeColMajorMatrixFile(argv[3], matArow, matBcol, matC);

    std::string output_file = std::string(argv[3]);

    std::string no_worksharing_output_file = output_file;
    replace(no_worksharing_output_file, ".txt", "_no_worksharing.txt");
    std::vector<float> noWorksharingC(matArow * matBcol);

    wall_clock = omp_get_wtime();
    sgemmNoWorksharing('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f,
                       &noWorksharingC.front(), matArow);
    time_elapsed = omp_get_wtime() - wall_clock;
    printf("no_worksharing: time_elapsed: %lf seconds\n", time_elapsed);
    writeColMajorMatrixFile(no_worksharing_output_file.c_str(), matArow, matBcol, noWorksharingC);

    if (validate_result(matC, noWorksharingC)) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }

    std::string worksharing_output_file = output_file;
    replace(worksharing_output_file, ".txt", "_worksharing.txt");
    std::vector<float> worksharingC(matArow * matBcol);

    wall_clock = omp_get_wtime();
    sgemmWorksharing('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f,
                     &worksharingC.front(), matArow);
    time_elapsed = omp_get_wtime() - wall_clock;
    printf("worksharing: time_elapsed: %lf seconds\n", time_elapsed);
    writeColMajorMatrixFile(worksharing_output_file.c_str(), matArow, matBcol, worksharingC);

    if (validate_result(matC, worksharingC)) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }

    std::string tasking_output_file = output_file;
    replace(tasking_output_file, ".txt", "_tasking.txt");
    std::vector<float> taskingC(matArow * matBcol);

    wall_clock = omp_get_wtime();
    sgemmTasking('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f,
                 &taskingC.front(), matArow);
    time_elapsed = omp_get_wtime() - wall_clock;
    printf("tasking: time_elapsed: %lf seconds\n", time_elapsed);
    writeColMajorMatrixFile(tasking_output_file.c_str(), matArow, matBcol, taskingC);

    if (validate_result(matC, taskingC)) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }

    return 0;
}
