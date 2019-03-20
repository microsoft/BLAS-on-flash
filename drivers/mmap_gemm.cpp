// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include "bof_types.h"
#include "mkl.h"
#include "utils.h"

using namespace std::chrono;

flash::Logger logger("in_mem");

int main(int argc, char** argv) {
  LOG_ASSERT(logger, argc == 15,
             "Usage Mode : <exec> <mat_A_file> <mat_B_file> <mat_C_file> "
             "<A_nrows> <A_ncols> <B_ncols> <alpha> <beta> <a transpose?> <b "
             "transpose?> <matr order> <lda_a> <lda_b> <lda_c>");

  // map matrices to flash pointers
  std::string A_name = std::string(argv[1]);
  std::string B_name = std::string(argv[2]);
  std::string C_name = std::string(argv[3]);
  // problem dimension
  FBLAS_UINT m = (FBLAS_UINT) std::stol(argv[4]);
  FBLAS_UINT k = (FBLAS_UINT) std::stol(argv[5]);
  FBLAS_UINT n = (FBLAS_UINT) std::stol(argv[6]);

  FPTYPE     alpha = (FPTYPE) std::stof(argv[7]);
  FPTYPE     beta = (FPTYPE) std::stof(argv[8]);
  CHAR       ta = argv[9][0];
  CHAR       tb = argv[10][0];
  CHAR       ord = argv[11][0];
  FBLAS_UINT lda_a = (FBLAS_UINT) std::stol(argv[12]);
  FBLAS_UINT lda_b = (FBLAS_UINT) std::stol(argv[13]);
  FBLAS_UINT lda_c = (FBLAS_UINT) std::stol(argv[14]);

  int A_fd = open(A_name.c_str(), O_RDONLY | O_DIRECT);
  assert(A_fd != -1);
  int B_fd = open(B_name.c_str(), O_RDONLY | O_DIRECT);
  assert(B_fd != -1);
  int C_fd = open(C_name.c_str(), O_RDWR | O_DIRECT);
  assert(C_fd != -1);

  float* mat_A = (float*) mmap(nullptr, m * k * sizeof(float), PROT_READ,
                               MAP_PRIVATE, A_fd, 0);
  assert(mat_A != (float*) MAP_FAILED);
  float* mat_B = (float*) mmap(nullptr, k * n * sizeof(float), PROT_READ,
                               MAP_PRIVATE, B_fd, 0);
  assert(mat_B != (float*) MAP_FAILED);
  float* mat_C = (float*) mmap(nullptr, m * n * sizeof(float), PROT_WRITE,
                               MAP_SHARED, C_fd, 0);
  assert(mat_C != (float*) MAP_FAILED);

  LOG_DEBUG(logger, "dimensions : A = ", m, "x", k, ", B = ", k, "x", n);
  LOG_INFO(logger, "Starting sgemm call");

  decltype(CblasNoTrans)  trans_a = ta == 'T' ? CblasTrans : CblasNoTrans;
  decltype(CblasNoTrans)  trans_b = tb == 'T' ? CblasTrans : CblasNoTrans;
  decltype(CblasRowMajor) mat_ord = ord == 'R' ? CblasRowMajor : CblasColMajor;
  // execute gemm call

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  mkl_gemm(mat_ord, trans_a, trans_b,          // ordering
           m, n, k,                            // sizes
           alpha, mat_A, lda_a, mat_B, lda_b,  // input
           beta, mat_C, lda_c);                // output
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> span = duration_cast<duration<double>>(t2 - t1);
  LOG_INFO(logger, "gemm() took ", span.count());

  int ret = munmap(mat_A, m * k * sizeof(float));
  assert(ret != -1);
  ret = munmap(mat_B, k * n * sizeof(float));
  assert(ret != -1);
  ret = munmap(mat_C, m * n * sizeof(float));
  assert(ret != -1);
  return 0;
}
