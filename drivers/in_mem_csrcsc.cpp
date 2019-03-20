// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstdio>
#include <fstream>
#include "bof_types.h"
#include "bof_utils.h"
#include "mkl.h"

flash::Logger logger("in_mem_csrcsc");

int main(int argc, char **argv) {
  LOG_ASSERT(logger, argc == 9,
             "Usage : <exec> <vals_a> <indices_a> <offsets_a> <vals_a_tr> "
             "<indices_a_tr> <offsets_a_tr> <n_rows> <n_cols>");

  std::string vals_a_name = std::string(argv[1]);
  std::string idxs_a_name = std::string(argv[2]);
  std::string offs_a_name = std::string(argv[3]);
  std::string vals_atr_name = std::string(argv[4]);
  std::string idxs_atr_name = std::string(argv[5]);
  std::string offs_atr_name = std::string(argv[6]);
  MKL_INT     m = (MKL_INT) std::stol(argv[7]);
  MKL_INT     n = (MKL_INT) std::stol(argv[8]);

  // echo program args to stdout
  LOG_INFO(logger, "Program arguments:");
  LOG_INFO(logger, "\t vals_a=", vals_a_name);
  LOG_INFO(logger, "\t idxs_a=", idxs_a_name);
  LOG_INFO(logger, "\t offs_a=", offs_a_name);
  LOG_INFO(logger, "\t vals_atr=", vals_atr_name);
  LOG_INFO(logger, "\t idxs_atr=", idxs_atr_name);
  LOG_INFO(logger, "\t offs_atr=", offs_atr_name);
  LOG_INFO(logger, "\t n_rows=", m);
  LOG_INFO(logger, "\t n_cols=", n);

  // read offs_a array first
  MKL_INT dim = std::max(m, n);
  LOG_INFO(logger, "Using dimension=", dim);
  MKL_INT *offs_a = new MKL_INT[dim + 1];
  MKL_INT *offs_atr = new MKL_INT[dim + 1];

  LOG_INFO(logger, "Reading offs_a from file");
  std::ifstream fs(offs_a_name, std::ios::binary);
  fs.read((char *) offs_a, (m + 1) * sizeof(MKL_INT));
  fs.close();

  // adjust offs_a array
  for (MKL_INT k = (m + 1); k <= dim; k++) {
    offs_a[k] = offs_a[m];
  }

  FBLAS_UINT nnzs = offs_a[dim] - offs_a[0];
  LOG_INFO(logger, "Will transpose nnzs=", nnzs, " values");

  MKL_INT *idxs_a = new MKL_INT[nnzs];
  MKL_INT *idxs_atr = new MKL_INT[nnzs];
  FPTYPE * vals_a = new FPTYPE[nnzs];
  FPTYPE * vals_atr = new FPTYPE[nnzs];

  // read a into memory
  LOG_INFO(logger, "Reading idxs_a from file");
  fs.open(idxs_a_name, std::ios::binary);
  fs.read((char *) idxs_a, nnzs * sizeof(MKL_INT));
  fs.close();
  LOG_INFO(logger, "Reading vals_a from file");
  fs.open(vals_a_name, std::ios::binary);
  fs.read((char *) vals_a, nnzs * sizeof(FPTYPE));
  fs.close();

  // setup MKL parameters
  MKL_INT job[6] = {0, 0, 0, -1, -1, 1};
  MKL_INT info = -1;  // not used

  // execute csrcsc call
  LOG_INFO(logger, "Starting csrcsc call");
  mkl_csrcsc(job, &dim, vals_a, idxs_a, offs_a, vals_atr, idxs_atr, offs_atr,
             &info);
  LOG_INFO(logger, "Finished csrcsc call");
  LOG_INFO(logger, "Input nnzs=", offs_a[dim], ", Output nnzs=", offs_atr[dim]);

  // write offsets to file
  LOG_INFO(logger, "Writing offs_aT to file=", offs_atr_name);
  std::ofstream ofs;
  ofs.open(offs_atr_name, std::ios::binary | std::ios::out);
  ofs.write((char *) offs_atr, (n + 1) * sizeof(MKL_INT));
  ofs.close();
  LOG_INFO(logger, "Writing idxs_aT to file=", idxs_atr_name);
  ofs.open(idxs_atr_name, std::ios::binary | std::ios::out);
  ofs.write((char *) idxs_atr, (nnzs) * sizeof(MKL_INT));
  ofs.close();
  LOG_INFO(logger, "Writing vals_aT to file=", vals_atr_name);
  ofs.open(vals_atr_name, std::ios::binary | std::ios::out);
  ofs.write((char *) vals_atr, (nnzs) * sizeof(FPTYPE));
  ofs.close();

  // free memory
  delete[] idxs_a;
  delete[] idxs_atr;
  delete[] offs_a;
  delete[] offs_atr;
  delete[] vals_a;
  delete[] vals_atr;
}
