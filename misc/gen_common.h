#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <errno.h>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <pthread.h>
using namespace std;

void check_file(int fd, const char *file, int64_t min_size)
{
    if (fd < 0) {
        printf("could not open file %s : %s\n", file, strerror(errno));
        exit(0);
    }

    int64_t fs = lseek(fd, 0, SEEK_END);
    if (fs < min_size) {
        printf("file too smaint64_t with only %int64_td bytes, file should be atleast %int64_td "
               "bytes\n",
               fs, min_size);
        exit(0);
    }
    lseek(fd, 0, SEEK_SET);
}

void check_mmap(void *d)
{
    if ((int64_t)d == -1) {
        printf("mmap error : %s\n", strerror(errno));
        exit(0);
    }
}

int64_t get_val(string s)
{
    int64_t mult = 1;
    switch (s[s.size() - 1]) {
        case 'G':
            mult *= 1024;
        case 'M':
            mult *= 1024;
        case 'K':
            mult *= 1024;

            s = s.substr(0, s.size() - 1);
            break;

        default:
            break;
    }

    return stoll(s) * mult;
}

void create_file(const char *file, int64_t size)
{
    FILE *fp = fopen(file, "w");
    fseek(fp, size - 1, SEEK_SET);
    fputc('\0', fp);
    fclose(fp);
}

inline double gflops(int64_t M, int64_t K, int64_t N, float dur)
{
    return ((double)(2 * M * N * K) / dur) / 1e9;
}

void init_cond(pthread_cond_t &x)
{
    if (pthread_cond_init(&x, NULL) != 0) {
        printf("error initialising condvar\n");
        exit(0);
    }
}
void init_mutex(pthread_mutex_t &x)
{
    if (pthread_mutex_init(&x, NULL) != 0) {
        printf("error initialising condvar\n");
        exit(0);
    }
}

void print_flops_info(const vector<float> &flops, int64_t M, int64_t K, int64_t N, int NUM_THR)
{
    float mean = 0, dev = 0;
    for (auto &d : flops)
        mean += d;
    mean /= (float)flops.size();

    for (auto &d : flops)
        dev += (d - mean) * (d - mean);
    dev /= (float)flops.size();
    dev = sqrt(dev);

    printf("%int64_tdx%int64_tdx%int64_td,%d,%.1f,%.1f\n", M, K, N, NUM_THR, mean, dev);
}

void print_matrix(float *a, int64_t M, int64_t N, string s)
{
    printf("\nMatrix %s\n", s.c_str());
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%.1f ", a[i * N + j]);
        printf("\n");
    }
}

int64_t round_up(int64_t in, int64_t max)
{
    int64_t t;
    for (int i = 2; i <= 16; i *= 2) {
        t = in + (in % i);
        if (t > max)
            break;
        in = t;
    }
    return in;
}

#endif
