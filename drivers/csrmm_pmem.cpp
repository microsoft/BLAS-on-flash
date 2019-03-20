// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <malloc.h>
#include "flash_blas.h"
#include "lib_funcs.h"
#include "bof_utils.h"

using namespace flash;

flash::Logger logger("csrmm");

int main(int argc, char** argv) {
  LOG_ASSERT(logger, argc == 13,
             "usage : <exec> <vals_A> <indices_A> <offsets_A> <vals_B> "
             "<vals_C> <A_nrows> <A_ncols> <B_ncols> <alpha> <beta> <trans_a> "
             "<ord_b>");
  // Extract problem parameters
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

  // map files
  LOG_INFO(logger, "Mapping files");
  flash_ptr<FPTYPE> a_vals_fptr =
      flash::map_file<FPTYPE>(a_vals, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> a_idxs_fptr =
      flash::map_file<MKL_INT>(a_idxs, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> a_offs_fptr =
      flash::map_file<MKL_INT>(a_offs, flash::Mode::READWRITE);

  // Read matrices B and C into memory
  FPTYPE* b_vals_ptr = new FPTYPE[a_ncols * b_ncols];
  LOG_INFO(logger, "Reading matrix B from file");
  std::ifstream fs(b_vals, std::ios::binary);
  fs.read((char*) b_vals_ptr, a_ncols * b_ncols * sizeof(FPTYPE));
  fs.close();
  LOG_INFO(logger, "Reading matrix C from file");
  FPTYPE* c_vals_ptr = (FPTYPE*) malloc(a_nrows * b_ncols * sizeof(FPTYPE));
  fs.open(c_vals, std::ios::binary);
  fs.read((char*) c_vals_ptr, a_nrows * b_ncols * sizeof(FPTYPE));
  fs.close();

  // execute csrmm call
  LOG_INFO(logger, "Starting csrmm call");
  csrmm(trans_a, a_nrows, a_ncols, b_ncols, alpha, beta, a_vals_fptr,
        a_offs_fptr, a_idxs_fptr, ord_b, b_vals_ptr, c_vals_ptr);
  LOG_INFO(logger, "Finished csrmm");
  LOG_INFO(logger, "malloc size for C=", malloc_usable_size(c_vals_ptr));

  // write results to disk
  LOG_INFO(logger, "Writing matrix C to file");
  std::fstream fs2(c_vals, std::ios::binary);
  fs2.write((char*) c_vals_ptr, a_nrows * b_ncols * sizeof(FPTYPE));
  fs2.close();

  // unmap files
  LOG_INFO(logger, "Unmapping files");
  unmap_file(a_vals_fptr);
  unmap_file(a_idxs_fptr);
  unmap_file(a_offs_fptr);

  // release memory
  LOG_INFO(logger, "Releasing memory");
  delete[] b_vals_ptr;
  free(c_vals_ptr);
  LOG_INFO(logger, "Exiting");
}