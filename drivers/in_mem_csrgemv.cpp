// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bof_types.h"
#include "flash_blas.h"
#include "lib_funcs.h"
#include "utils.h"

using namespace flash;
flash::Logger logger("csrgemv");

int main(int argc, char **argv) {
  LOG_ASSERT(logger, argc == 9,
             "usage : <exec> <vals_A> <indices_A> <offsets_A> <vals_B> "
             "<vals_C> <A_nrows> <A_ncols> <trans_a>");

  // Extract problem parameters
  std::string a_vals(argv[1]);
  std::string a_idxs(argv[2]);
  std::string a_offs(argv[3]);
  std::string b_vals(argv[4]);
  std::string c_vals(argv[5]);
  FBLAS_UINT  a_nrows = std::stol(argv[6]);
  FBLAS_UINT  a_ncols = std::stol(argv[7]);
  CHAR        trans_a = argv[8][0];
  LOG_INFO(logger, "Program Arguments:");
  LOG_INFO(logger, "a_vals=", a_vals);
  LOG_INFO(logger, "a_idxs=", a_idxs);
  LOG_INFO(logger, "a_offs=", a_offs);
  LOG_INFO(logger, "b_vals=", b_vals);
  LOG_INFO(logger, "c_vals=", c_vals);
  LOG_INFO(logger, "a_nrows=", a_nrows);
  LOG_INFO(logger, "a_ncols=", a_ncols);
  LOG_INFO(logger, "trans_a=", trans_a);

  // compute dimension
  MKL_INT dim = std::max(a_nrows, a_ncols);

  // read files into memory
  MKL_INT *a_offs_ptr = new MKL_INT[dim + 1];
  LOG_INFO(logger, "Reading a_offs from file");
  std::ifstream fs(a_offs, std::ios::binary);
  fs.read((char *) a_offs_ptr, (a_nrows + 1) * sizeof(MKL_INT));
  fs.close();
  for (MKL_INT i = a_nrows + 1; i <= dim; i++) {
    a_offs_ptr[i] = a_offs_ptr[a_nrows];
  }

  FBLAS_UINT nnzs = a_offs_ptr[a_nrows] - a_offs_ptr[0];
  LOG_INFO(logger, "Using nnzs=", nnzs);
  FPTYPE *a_vals_ptr = new FPTYPE[nnzs];
  LOG_INFO(logger, "Reading a_vals from file");
  fs.open(a_vals, std::ios::binary);
  fs.read((char *) a_vals_ptr, nnzs * sizeof(FPTYPE));
  fs.close();

  MKL_INT *a_idxs_ptr = new MKL_INT[nnzs];
  LOG_INFO(logger, "Reading a_idxs from file");
  fs.open(a_idxs, std::ios::binary);
  fs.read((char *) a_idxs_ptr, nnzs * sizeof(MKL_INT));
  fs.close();

  FBLAS_UINT b_len = (trans_a == 'N' ? a_ncols : a_nrows);
  FBLAS_UINT c_len = (trans_a == 'N' ? a_nrows : a_ncols);
  FPTYPE *   b_vals_ptr = new FPTYPE[dim];
  FPTYPE *   c_vals_ptr = new FPTYPE[dim];
  memset(b_vals_ptr, 0, dim * sizeof(FPTYPE));
  memset(c_vals_ptr, 0, dim * sizeof(FPTYPE));

  LOG_INFO(logger, "Reading vector b from file");
  fs.open(b_vals, std::ios::binary);
  fs.read((char *) b_vals_ptr, b_len * sizeof(FPTYPE));
  fs.close();

  LOG_INFO(logger, "Reading vector c from file");
  fs.open(c_vals, std::ios::binary);
  fs.read((char *) c_vals_ptr, c_len * sizeof(FPTYPE));
  fs.close();

  // execute csrmm call
  LOG_INFO(logger, "Starting mkl_csrgemv call");
  mkl_csrgemv(&trans_a, &dim, a_vals_ptr, a_offs_ptr, a_idxs_ptr, b_vals_ptr,
              c_vals_ptr);
  LOG_INFO(logger, "Finished mkl_csrgemv");

  // unmap files
  LOG_INFO(logger, "Unmapping files");
  LOG_INFO(logger, "Writing vector c to file");
  std::ofstream fs2(c_vals, std::ios::binary);
  fs2.write((char *) c_vals_ptr, c_len * sizeof(FPTYPE));
  fs2.close();

  LOG_INFO(logger, "Releasing memory");
  delete[] a_offs_ptr;
  delete[] a_idxs_ptr;
  delete[] a_vals_ptr;
  delete[] b_vals_ptr;
  delete[] c_vals_ptr;

  LOG_INFO(logger, "Exiting");
}