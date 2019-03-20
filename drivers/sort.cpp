// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bof_types.h"
#include "bof_utils.h"
#include "flash_blas.h"
#include "lib_funcs.h"

using namespace flash;
flash::Logger logger("sort");

int main(int argc, char **argv) {
  LOG_ASSERT(logger, argc == 4, "usage : <exec> <in_file> <out_file> <size>");

  // Extract problem parameters
  std::string in_fname(argv[1]);
  std::string out_fname(argv[2]);
  FBLAS_UINT  size = (FBLAS_UINT) std::stol(argv[3]);
  LOG_INFO(logger, "Program Arguments:");
  LOG_INFO(logger, "in_fname=", in_fname);
  LOG_INFO(logger, "out_fname=", out_fname);
  LOG_INFO(logger, "size=", size);

  // map files
  LOG_INFO(logger, "Mapping files");
  flash_ptr<FBLAS_UINT> in_fptr =
      flash::map_file<FBLAS_UINT>(in_fname, flash::Mode::READWRITE);
  flash_ptr<FBLAS_UINT> out_fptr =
      flash::map_file<FBLAS_UINT>(out_fname, flash::Mode::READWRITE);

  // execute sort call
  LOG_INFO(logger, "Starting sort call");
  flash::sort<FBLAS_UINT>(in_fptr, out_fptr, size);
  LOG_INFO(logger, "Finished sort");

  // unmap files
  LOG_INFO(logger, "Unmapping files");
  flash::unmap_file(in_fptr);
  flash::unmap_file(out_fptr);

  LOG_INFO(logger, "Exiting");
}
