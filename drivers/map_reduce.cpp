// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cmath>
#include "flash_blas.h"
#include "lib_funcs.h"
#include "utils.h"

using namespace flash;

flash::Logger logger("map-reduce");

int main(int argc, char **argv) {
  LOG_ASSERT(logger, argc == 4,
             "Computes B(i)=sqrt(A(i)) and prints sum(B(i))"
             "usage : <exec> <A_vals> <B_vals> <n_vals>");

  // Extract problem parameters
  std::string a_vals(argv[1]);
  std::string b_vals(argv[2]);
  FBLAS_UINT  n_vals = (FBLAS_UINT) std::stol(argv[3]);
  LOG_INFO(logger, "Program Arguments:");
  LOG_INFO(logger, "a_vals=", a_vals);
  LOG_INFO(logger, "b_vals=", b_vals);
  LOG_INFO(logger, "n_vals=", n_vals);

  // map files
  LOG_INFO(logger, "Mapping files");
  flash_ptr<FBLAS_UINT> a_vals_fptr =
      flash::map_file<FBLAS_UINT>(a_vals, flash::Mode::READWRITE);
  flash_ptr<FPTYPE> b_vals_fptr =
      flash::map_file<FPTYPE>(b_vals, flash::Mode::READWRITE);

  // execute map call
  LOG_INFO(logger, "Starting map call");
  std::function<FPTYPE(const FBLAS_UINT &)> map_fn =
      [&](const FBLAS_UINT &in_val) { return std::sqrt(in_val); };
  FBLAS_INT                 ret = map(a_vals_fptr, b_vals_fptr, n_vals, map_fn);
  LOG_INFO(logger, "Finished map with return_val=", ret);

  // execute reduce call
  std::function<FPTYPE(FPTYPE &, FPTYPE &)> reduce_fn = [&](
      FPTYPE &l, FPTYPE &r) { return l + r; };
  LOG_INFO(logger, "Starting reduce call");
  FPTYPE id = 0.0f;
  FPTYPE reduce_result = reduce(b_vals_fptr, n_vals, id, reduce_fn);
  LOG_INFO(logger, "Finished reduce with return=", reduce_result);
  std::printf("\nResult=%15.5f", reduce_result);

  // unmap files
  LOG_INFO(logger, "Unmapping files");
  unmap_file(a_vals_fptr);
  unmap_file(b_vals_fptr);

  LOG_INFO(logger, "Exiting");
}