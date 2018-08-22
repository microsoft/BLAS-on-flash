// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "flash_blas.h"
#include "lib_funcs.h"
#include "scheduler/scheduler.h"
#include "utils.h"

using namespace flash;

flash::Logger logger("csrcsc");

namespace flash {
  extern Scheduler sched;
}

int main(int argc, char** argv) {
  LOG_ASSERT(logger, argc == 9,
             "Usage : <exec> <vals_a> <indices_a> <offsets_a> <vals_a_tr> "
             "<indices_a_tr> <offsets_a_tr> <n_rows> <n_cols>");
  // Extract problem parameters
  std::string a_vals(argv[1]);
  std::string a_idxs(argv[2]);
  std::string a_offs(argv[3]);
  std::string atr_vals(argv[4]);
  std::string atr_idxs(argv[5]);
  std::string atr_offs(argv[6]);
  FBLAS_UINT  n_rows = std::stol(argv[7]);
  FBLAS_UINT  n_cols = std::stol(argv[8]);
  LOG_INFO(logger, "Program Arguments:");
  LOG_INFO(logger, "a_vals=", a_vals);
  LOG_INFO(logger, "a_idxs=", a_idxs);
  LOG_INFO(logger, "a_offs=", a_offs);
  LOG_INFO(logger, "atr_vals=", atr_vals);
  LOG_INFO(logger, "atr_idxs=", atr_idxs);
  LOG_INFO(logger, "atr_offs=", atr_offs);
  LOG_INFO(logger, "a_nrows=", n_rows);
  LOG_INFO(logger, "a_ncols=", n_cols);
  LOG_INFO(logger, "Setting up flash context");
  flash::flash_setup("/raid/tmp/");

  LOG_INFO(logger, "Mapping files");
  // map files
  flash_ptr<FPTYPE> a_vals_fptr =
      flash::map_file<FPTYPE>(a_vals, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> a_idxs_fptr =
      flash::map_file<MKL_INT>(a_idxs, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> a_offs_fptr =
      flash::map_file<MKL_INT>(a_offs, flash::Mode::READWRITE);
  flash_ptr<FPTYPE> atr_vals_fptr =
      flash::map_file<FPTYPE>(atr_vals, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> atr_idxs_fptr =
      flash::map_file<MKL_INT>(atr_idxs, flash::Mode::READWRITE);
  flash_ptr<MKL_INT> atr_offs_fptr =
      flash::map_file<MKL_INT>(atr_offs, flash::Mode::READWRITE);
  LOG_INFO(logger, "Starting csrcsc call");
  // execute csrcsc call
  flash::sched.set_num_compute_threads(8);
  csrcsc(n_rows, n_cols, a_offs_fptr, a_idxs_fptr, a_vals_fptr, atr_offs_fptr,
         atr_idxs_fptr, atr_vals_fptr);
  flash::sched.set_num_compute_threads(1);
  LOG_INFO(logger, "Finished csrcsc");

  LOG_INFO(logger, "Unmapping files");
  // unmap files
  unmap_file(a_vals_fptr);
  unmap_file(a_idxs_fptr);
  unmap_file(a_offs_fptr);
  unmap_file(atr_vals_fptr);
  unmap_file(atr_idxs_fptr);
  unmap_file(atr_offs_fptr);
  LOG_INFO(logger, "Destroying flash context");
  flash::flash_destroy();
  LOG_INFO(logger, "Exiting");
}
