// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "flash_blas.h"
#include "lib_funcs.h"
#include "types.h"
#include "utils.h"

using namespace flash;
flash::Logger logger("in_mem_sort");

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

  // read files into memory
  FBLAS_UINT *in_ptr = new FBLAS_UINT[size];
  LOG_INFO(logger, "Reading input from file");
  std::ifstream fs(in_fname, std::ios::binary);
  fs.read((char *) in_ptr, (size) * sizeof(FBLAS_UINT));
  fs.close();

  // execute parallel sort
  LOG_INFO(logger, "Starting sort call");
  __gnu_parallel::sort(in_ptr, in_ptr + size);
  LOG_INFO(logger, "Finished sort call");

  // dump output to file
  LOG_INFO(logger, "Writing output to file");
  std::ofstream fs2(out_fname, std::ios::binary);
  fs2.write((char *) in_ptr, (size) * sizeof(FBLAS_UINT));
  fs2.close();

  // unmap files
  LOG_INFO(logger, "Freeing memory");
  delete[] in_ptr;

  LOG_INFO(logger, "Exiting");
}
