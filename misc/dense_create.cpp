#include "gen_common.h"
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("usage : %s <filename> <nrows> <ncols> <fill_mode>\n", argv[0]);
    exit(0);
  }

  int64_t MAT_SIZE = stoll(argv[2]) * stoll(argv[3]) * sizeof(float);

  create_file(argv[1], MAT_SIZE);
  int fd = open(argv[1], O_RDWR);
  check_file(fd, argv[1], MAT_SIZE);
  float *c =
      (float *) mmap(NULL, MAT_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  check_mmap(c);

  srand(time(NULL));
  int64_t N_ELEMENTS = MAT_SIZE / 4;
  if (argv[4][0] == 'r') {
    unsigned int r = 0;
#pragma omp      parallel for schedule(static)
    for (int64_t i = 0; i < N_ELEMENTS; i++) {
      c[i] = (i + rand_r(&r)) % 10;
    }
    // rand() seems to have a mutex, so it is slow with threads.
  } else if (argv[4][0] == 's') {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N_ELEMENTS; i++) {
      c[i] = i % 10;
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N_ELEMENTS; i++) {
      c[i] = 0;
    }
  }

  munmap(c, MAT_SIZE);

  return 0;
}
