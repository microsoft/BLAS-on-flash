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

  // map files
  LOG_INFO(logger, "Mapping files");
  flash_ptr<FPTYPE> a_vals_fptr =
      flash::map_file<FPTYPE>(a_vals, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> a_idxs_fptr =
      flash::map_file<MKL_INT>(a_idxs, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> a_offs_fptr =
      flash::map_file<MKL_INT>(a_offs, flash::Mode::READWRITE);
  // read files into memory
  FBLAS_UINT b_len = (trans_a == 'N' ? a_ncols : a_nrows);
  FBLAS_UINT c_len = (trans_a == 'N' ? a_nrows : a_ncols);
  FPTYPE *   b_valptr = new FPTYPE[b_len];
  FPTYPE *   c_valptr = new FPTYPE[c_len];

  LOG_INFO(logger, "Reading vector b from file");
  std::ifstream fs(b_vals, std::ios::binary);
  fs.read((char *) b_valptr, b_len * sizeof(FPTYPE));
  fs.close();

  LOG_INFO(logger, "Reading vector c from file");
  fs.open(c_vals, std::ios::binary);
  fs.read((char *) c_valptr, c_len * sizeof(FPTYPE));
  fs.close();

  // execute csrmm call
  LOG_INFO(logger, "Starting csrgemv call");
  csrgemv(trans_a, a_nrows, a_ncols, a_vals_fptr, a_offs_fptr, a_idxs_fptr,
          b_valptr, c_valptr);
  LOG_INFO(logger, "Finished csrgemv");

  // unmap files
  LOG_INFO(logger, "Unmapping files");
  unmap_file(a_vals_fptr);
  unmap_file(a_idxs_fptr);
  unmap_file(a_offs_fptr);
  LOG_INFO(logger, "Writing vector c to file");
  std::ofstream fs2(c_vals, std::ios::binary);
  fs2.write((char *) c_valptr, c_len * sizeof(FPTYPE));
  fs2.close();

  LOG_INFO(logger, "Releasing memory");
  delete[] b_valptr;
  delete[] c_valptr;

  LOG_INFO(logger, "Exiting");
}