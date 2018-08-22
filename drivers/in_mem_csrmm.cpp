// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "flash_blas.h"
#include "lib_funcs.h"
#include "timer.h"
#include "types.h"
#include "utils.h"

using namespace flash;
flash::Logger logger("in_mem");

int main(int argc, char **argv) {
  LOG_ASSERT(logger, argc == 13,
             "usage : <exec> <vals_A> <indices_A> <offsets_A> <vals_B> "
             "<vals_C> <A_nrows> <A_ncols> <B_ncols> <alpha> <beta> <trans_a> "
             "<ord_b>");

  std::string a_vals(argv[1]);
  std::string a_idxs(argv[2]);
  std::string a_offs(argv[3]);
  std::string b_vals(argv[4]);
  std::string c_vals(argv[5]);
  FBLAS_UINT  a_nrows = std::stol(argv[6]);
  FBLAS_UINT  a_ncols = std::stol(argv[7]);
  FBLAS_UINT  b_ncols = std::stol(argv[8]);
  FPTYPE      alpha = std::stof(argv[9]);
  FPTYPE      beta = std::stof(argv[10]);
  CHAR        trans_a = argv[11][0];
  CHAR        ord_b = argv[12][0];
  LOG_INFO(logger, "Program Arguments:");
  LOG_INFO(logger, "a_vals=", a_vals);
  LOG_INFO(logger, "a_idxs=", a_idxs);
  LOG_INFO(logger, "a_offs=", a_offs);
  LOG_INFO(logger, "b_vals=", b_vals);
  LOG_INFO(logger, "c_vals=", c_vals);
  LOG_INFO(logger, "a_nrows=", a_nrows);
  LOG_INFO(logger, "a_ncols=", a_ncols);
  LOG_INFO(logger, "b_ncols=", b_ncols);
  LOG_INFO(logger, "alpha=", alpha);
  LOG_INFO(logger, "beta=", beta);
  LOG_INFO(logger, "trans_a=", trans_a);
  LOG_INFO(logger, "ord_b=", ord_b);

  // read offs_a array first
  MKL_INT *offs_a = new MKL_INT[a_nrows + 1];

  LOG_INFO(logger, "Reading offs_a from file");
  std::ifstream fs(a_offs, std::ios::binary);
  fs.read((char *) offs_a, (a_nrows + 1) * sizeof(MKL_INT));
  fs.close();

  FBLAS_UINT nnzs = offs_a[a_nrows] - offs_a[0];
  LOG_INFO(logger, "Using nnzs=", nnzs);

  MKL_INT *idxs_a = new MKL_INT[nnzs];
  FPTYPE * vals_a = new FPTYPE[nnzs];

  // read a into memory
  LOG_INFO(logger, "Reading idxs_a from file");
  fs.open(a_idxs, std::ios::binary);
  fs.read((char *) idxs_a, nnzs * sizeof(MKL_INT));
  fs.close();
  LOG_INFO(logger, "Reading vals_a from file");
  fs.open(a_vals, std::ios::binary);
  fs.read((char *) vals_a, nnzs * sizeof(FPTYPE));
  fs.close();

  // read b into memory
  FPTYPE *vals_b;
  FPTYPE *vals_c;
  if (trans_a == 'N') {
    LOG_INFO(logger, "Reading vals_b from file");
    vals_b = new FPTYPE[a_ncols * b_ncols];
    fs.open(b_vals, std::ios::binary);
    fs.read((char *) vals_b, a_ncols * b_ncols * sizeof(FPTYPE));
    fs.close();
    LOG_INFO(logger, "Reading vals_c from file");
    vals_c = new FPTYPE[a_nrows * b_ncols];
    fs.open(c_vals, std::ios::binary);
    fs.read((char *) vals_c, a_nrows * b_ncols * sizeof(FPTYPE));
    fs.close();
  } else {
    LOG_INFO(logger, "Reading vals_b from file");
    vals_b = new FPTYPE[a_nrows * b_ncols];
    fs.open(b_vals, std::ios::binary);
    fs.read((char *) vals_b, a_nrows * b_ncols * sizeof(FPTYPE));
    fs.close();
    LOG_INFO(logger, "Reading vals_c from file");
    vals_c = new FPTYPE[a_ncols * b_ncols];
    fs.open(c_vals, std::ios::binary);
    fs.read((char *) vals_c, a_ncols * b_ncols * sizeof(FPTYPE));
    fs.close();
  }

  // prepare for MKL call
  MKL_INT m = (MKL_INT) a_nrows;
  MKL_INT n = (MKL_INT) b_ncols;
  MKL_INT k = (MKL_INT) a_ncols;
  char    matdescra[4] = {'G', 'X', 'X', (ord_b == 'C' ? 'F' : 'C')};
  if (ord_b == 'C') {
#pragma omp parallel for
    for (FBLAS_UINT i = 1; i <= a_nrows; i++) {
      offs_a[i] = offs_a[i] - offs_a[0] + 1;
    }
    offs_a[0] = 1;
#pragma omp parallel for
    for (FBLAS_UINT j = 0; j < nnzs; j++) {
      idxs_a[j]++;
    }
  }

  MKL_INT ldb = (ord_b == 'C' ? k : n);
  MKL_INT ldc = (ord_b == 'C' ? m : n);

  // start MKL call
  LOG_INFO(logger, "Starting mkl_csrmm call");
  Timer timer;
  mkl_csrmm(&trans_a, &m, &n, &k, &alpha, &matdescra[0], vals_a, idxs_a, offs_a,
            offs_a + 1, vals_b, &ldb, &beta, vals_c, &ldc);
  LOG_INFO(logger, "mkl_csrmm() took ", timer.elapsed() / 1000);
  LOG_INFO(logger, "Finished mkl_csrmm call");

  // write C back to storage
  LOG_INFO(logger, "Write vals_c to file");
  std::ofstream fs2;
  fs2.open(c_vals, std::ios::binary);
  fs2.write((char *) vals_c,
            (trans_a == 'N' ? a_nrows : a_ncols) * b_ncols * sizeof(FPTYPE));
  fs2.close();

  // cleanup
  LOG_INFO(logger, "Cleaning up");
  delete[] offs_a;
  delete[] idxs_a;
  delete[] vals_a;
  delete[] vals_b;
  delete[] vals_c;

  LOG_INFO(logger, "exiting");
}
