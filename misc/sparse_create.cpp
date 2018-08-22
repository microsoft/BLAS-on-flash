#include "mkl.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include "gen_common.h"
using namespace std;

typedef float FPTYPE;
double        sparsity;
MKL_INT       nrows, ncols, nnz_per_row, nnz;

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("usage : %s <name> <nrows> <ncols> <sparsity>\n", argv[0]);
    exit(0);
  }

  string name = string(argv[1]);
  nrows = stoll(argv[2]);
  ncols = stoll(argv[3]);
  sparsity = stod(argv[4]);
  assert(sparsity < 1.0f);
  nnz_per_row = ceil(ncols * sparsity);
  nnz = nrows * nnz_per_row;

  ofstream myfile;
  string info_name(name + "info");
  string csr_name(name + "csr");
  string col_name(name + "col");
  string off_name(name + "off");
  myfile.open(info_name);
  myfile << nrows << " " << ncols << " " << sparsity << "\n";
  myfile.close();

  create_file(off_name.c_str(), sizeof(MKL_INT) * (nrows + 1));
  create_file(csr_name.c_str(), sizeof(FPTYPE) * nnz);
  create_file(col_name.c_str(), sizeof(MKL_INT) * nnz);

  int fd = open(off_name.c_str(), O_RDWR);
  check_file(fd, off_name.c_str(), sizeof(MKL_INT) * (nrows + 1));
  MKL_INT *off = (MKL_INT *) mmap(NULL, sizeof(MKL_INT) * (nrows + 1),
                                  PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  check_mmap(off);

  fd = open(csr_name.c_str(), O_RDWR);
  check_file(fd, csr_name.c_str(), sizeof(FPTYPE) * nnz);
  FPTYPE *csr = (FPTYPE *) mmap(NULL, sizeof(FPTYPE) * nnz,
                                PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  check_mmap(csr);

  srand(time(0));
  #pragma omp parallel for schedule(static)
  for(MKL_INT i = 0; i < nnz; i++) {
    csr[i] = (FPTYPE)((i % 9) + 1);
  }

  fd = open(col_name.c_str(), O_RDWR);
  check_file(fd, col_name.c_str(), sizeof(MKL_INT) * nnz);
  MKL_INT *col = (MKL_INT *) mmap(NULL, sizeof(MKL_INT) * nnz,
                                  PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  check_mmap(col);

#pragma omp parallel for schedule(static)
  for (MKL_INT r = 0; r < nrows; r++) {
    off[r] = r * nnz_per_row;

    unsigned int seed = r;

    vector<MKL_INT> col_vec(nnz_per_row + 40);
    for (MKL_INT i = 0; i < col_vec.size(); i++)
      col_vec[i] =
          ((rand_r(&seed) + ((MKL_INT) rand_r(&seed)) * RAND_MAX) % ncols);
    sort(col_vec.begin(), col_vec.end());

    col_vec.erase(unique(col_vec.begin(), col_vec.end()), col_vec.end());

    assert(col_vec.size() >= nnz_per_row);
    for (MKL_INT i = 0; i < nnz_per_row; ++i)
      col[r * nnz_per_row + i] = col_vec[i];
  }
  off[nrows] = nnz;

  munmap(off, sizeof(MKL_INT) * (nrows + 1));
  munmap(csr, sizeof(FPTYPE) * nnz);
  munmap(col, sizeof(MKL_INT) * nnz);

  return 0;
}
