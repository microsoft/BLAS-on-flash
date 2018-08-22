// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cassert>
#include <chrono>
#include "flash_blas.h"
#include "lib_funcs.h"
#include "utils.h"

using namespace std::chrono;

std::string mnt_dir = "/tmp/gemm_driver_temps";

flash::Logger logger("gemm_driver");

int main(int argc, char** argv) {
  if (argc != 15) {
    LOG_INFO(logger,
             "Usage Mode : <exec> <mat_A_file> <mat_B_file> <mat_C_file> "
             "<A_nrows> <A_ncols> <B_ncols> <alpha> <beta> <a transpose?> <b "
             "transpose?> <matr order> <lda_a> <lda_b> <lda_c>");
    LOG_FATAL(logger, "expected 14 args, got ", argc - 1);
  }

  // init blas-on-flash
  LOG_DEBUG(logger, "setting up flash context");
  flash::flash_setup(mnt_dir);

  // map matrices to flash pointers
  std::string A_name = std::string(argv[1]);
  std::string B_name = std::string(argv[2]);
  std::string C_name = std::string(argv[3]);
  LOG_DEBUG(logger, "map matrices to flash_ptr");
  flash::flash_ptr<FPTYPE> mat_A =
      flash::map_file<FPTYPE>(A_name, flash::Mode::READWRITE);
  flash::flash_ptr<FPTYPE> mat_B =
      flash::map_file<FPTYPE>(B_name, flash::Mode::READWRITE);
  flash::flash_ptr<FPTYPE> mat_C =
      flash::map_file<FPTYPE>(C_name, flash::Mode::READWRITE);

  // problem dimension
  FBLAS_UINT m = (FBLAS_UINT) std::stol(argv[4]);
  FBLAS_UINT k = (FBLAS_UINT) std::stol(argv[5]);
  FBLAS_UINT n = (FBLAS_UINT) std::stol(argv[6]);
  FPTYPE     alpha = (FPTYPE) std::stof(argv[7]);
  FPTYPE     beta = (FPTYPE) std::stof(argv[8]);
  CHAR       trans_a = argv[9][0];
  CHAR       trans_b = argv[10][0];
  CHAR       mat_ord = argv[11][0];
  FBLAS_UINT lda_a = (FBLAS_UINT) std::stol(argv[12]);
  FBLAS_UINT lda_b = (FBLAS_UINT) std::stol(argv[13]);
  FBLAS_UINT lda_c = (FBLAS_UINT) std::stol(argv[14]);

  LOG_INFO(logger, "dimensions : A = ", m, "x", k, ", B = ", k, "x", n);

  // execute gemm call
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  FBLAS_INT res = flash::gemm(mat_ord, trans_a, trans_b, m, n, k, alpha, beta,
                              mat_A, mat_B, mat_C, lda_a, lda_b, lda_c);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> span = duration_cast<duration<double>>(t2 - t1);
  LOG_INFO(logger, "gemm() took ", span.count());

  LOG_INFO(logger, "flash::gemm() returned with ", res);

  LOG_DEBUG(logger, "un-map matrices");
  // unmap files
  flash::unmap_file(mat_A);
  flash::unmap_file(mat_B);
  flash::unmap_file(mat_C);

  LOG_DEBUG(logger, "destroying flash context");
  flash::flash_destroy();
  // destroy blas-on-flash
}
