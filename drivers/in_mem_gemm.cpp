// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <chrono>
#include <fstream>
#include "bof_types.h"
#include "mkl.h"
#include "bof_utils.h"

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

  float* mat_A = (float*) mkl_malloc(m * k * sizeof(float), 4096);
  float* mat_B = (float*) mkl_malloc(n * k * sizeof(float), 4096);
  float* mat_C = (float*) mkl_malloc(m * n * sizeof(float), 4096);

  LOG_INFO(logger, "Reading matrix A into memory");
  std::ifstream a_file(A_name, std::ios::binary);
  a_file.read((char*) mat_A, m * k * sizeof(float));
  a_file.close();
  LOG_INFO(logger, "Reading matrix B into memory");
  std::ifstream b_file(B_name, std::ios::binary);
  b_file.read((char*) mat_B, k * n * sizeof(float));
  b_file.close();
  LOG_INFO(logger, "Reading matrix C into memory");
  std::ifstream c_file(C_name, std::ios::binary);
  c_file.read((char*) mat_C, m * n * sizeof(float));
  c_file.close();

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

  LOG_INFO(logger, "Writing C to file");
  std::ofstream cout_file(C_name, std::ios::binary);
  cout_file.write((char*) mat_C, m * n * sizeof(float));
  cout_file.close();

  // free memory
  mkl_free(mat_A);
  mkl_free(mat_B);
  mkl_free(mat_C);

  return 0;
}
