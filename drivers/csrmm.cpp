// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bof_timer.h"
#include "bof_utils.h"
#include "flash_blas.h"
#include "lib_funcs.h"

using namespace flash;

int main(int argc, char** argv) {
  GLOG_ASSERT(argc == 13,
              "usage : <exec> <vals_A> <indices_A> <offsets_A> <vals_B> "
              "<vals_C> <A_nrows> <A_ncols> <B_ncols> <alpha> <beta> <trans_a> "
              "<ord_b>");

  // Set up flash context
  flash::flash_setup("");

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
  GLOG_INFO("Program Arguments:");
  GLOG_INFO("a_vals=", a_vals);
  GLOG_INFO("a_idxs=", a_idxs);
  GLOG_INFO("a_offs=", a_offs);
  GLOG_INFO("b_vals=", b_vals);
  GLOG_INFO("c_vals=", c_vals);
  GLOG_INFO("a_nrows=", a_nrows);
  GLOG_INFO("a_ncols=", a_ncols);
  GLOG_INFO("b_ncols=", b_ncols);
  GLOG_INFO("alpha=", alpha);
  GLOG_INFO("beta=", beta);
  GLOG_INFO("trans_a=", trans_a);
  GLOG_INFO("ord_b=", ord_b);

  // map files
  GLOG_INFO("Mapping files");
  flash_ptr<FPTYPE> a_vals_fptr =
      flash::map_file<FPTYPE>(a_vals, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> a_idxs_fptr =
      flash::map_file<MKL_INT>(a_idxs, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> a_offs_fptr =
      flash::map_file<MKL_INT>(a_offs, flash::Mode::READWRITE);
  flash_ptr<FPTYPE> b_vals_fptr =
      flash::map_file<FPTYPE>(b_vals, flash::Mode::READWRITE);
  flash_ptr<FPTYPE> c_vals_fptr =
      flash::map_file<FPTYPE>(c_vals, flash::Mode::READWRITE);

  // execute csrmm call
  GLOG_INFO("Starting csrmm call");
  Timer timer;
  csrmm(trans_a, a_nrows, a_ncols, b_ncols, alpha, beta, a_vals_fptr,
        a_offs_fptr, a_idxs_fptr, ord_b, b_vals_fptr, c_vals_fptr);
  GLOG_INFO("csrmm() took ", timer.elapsed() / 1000);
  GLOG_INFO("Finished csrmm");

  // unmap files
  GLOG_INFO("Unmapping files");
  unmap_file(a_vals_fptr);
  unmap_file(a_idxs_fptr);
  unmap_file(a_offs_fptr);
  unmap_file(b_vals_fptr);
  unmap_file(c_vals_fptr);
  GLOG_INFO("Exiting");
  flash::flash_destroy();
}
