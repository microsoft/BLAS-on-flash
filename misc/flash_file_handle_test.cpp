// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "file_handles/flash_file_handle.h"
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include "types.h"

#define N_TESTS 1000

using namespace flash;
namespace {
  void default_callback() {
  }
  std::function<void(void)> callback_fn = default_callback;

  bool verify_iota(FBLAS_UINT* buf, FBLAS_UINT offset, FBLAS_UINT buf_size) {
    FBLAS_UINT        n_vals = buf_size / sizeof(FBLAS_UINT);
    FBLAS_UINT        val_begin = offset / sizeof(FBLAS_UINT);
    std::atomic<bool> fail(false);

#pragma omp parallel for
    for (FBLAS_UINT i = 0; i < n_vals; i++) {
      if (buf[i] != (val_begin + i)) {
      }
    }
    return !fail.load();
  }

  bool verify_iota_stride(FBLAS_UINT* buf, FBLAS_UINT offset,
                          StrideInfo sinfo) {
    // verify each stride
    FBLAS_UINT        n_vals = sinfo.len_per_stride / sizeof(FBLAS_UINT);
    std::atomic<bool> fail(false);

#pragma omp parallel for
    for (FBLAS_UINT s = 0; s < sinfo.n_strides; s++) {
      FBLAS_UINT c_offset = offset + (s * sinfo.stride);
      FBLAS_UINT val_begin = c_offset / sizeof(FBLAS_UINT);
      // verify each value in stride
      for (FBLAS_UINT i = 0; i < n_vals; i++) {
        if (buf[(s * n_vals) + i] != (val_begin + i)) {
          fail.store(true);
        }
      }
    }
    return !fail.load();
  }

  void create_file(std::string& fname, FBLAS_UINT size) {
    std::ofstream           fout;
    std::vector<FBLAS_UINT> vals(size / sizeof(FBLAS_UINT));
    std::iota(vals.begin(), vals.end(), 0);
    fout.open(fname, std::ios::out | std::ios::binary);
    fout.write((char*) vals.data(), size * sizeof(FBLAS_UINT));
    fout.flush();
    fout.close();
  }
}  // namespace

void test_read(FlashFileHandle& fhandle, FBLAS_UINT fsize,
               FBLAS_UINT max_buf_size) {
  FBLAS_UINT* buf = new FBLAS_UINT[max_buf_size / sizeof(FBLAS_UINT)];
  FBLAS_UINT  max_read_offset = ROUND_DOWN(fsize - max_buf_size, 8);
  GLOG_ASSERT(max_read_offset > 8, "file size too small OR bad buffer size");

  FBLAS_UINT n_pass = 0;
  for (FBLAS_UINT i = 0; i < N_TESTS; i++) {
    FBLAS_UINT offset = ROUND_UP(rand() % max_read_offset, 8);
    FBLAS_UINT len = ROUND_UP(rand() % max_buf_size, 8);
    len = (len > 0 ? len : 128);
    fhandle.read(offset, len, buf, callback_fn);
    GLOG_INFO("Contiguous Read test #", i + 1, ": offset=", offset,
              ", length=", len);
    if (!verify_iota(buf, offset, len)) {
      GLOG_FAIL("Contiguous Write test #", i + 1, " failed");
    } else {
      n_pass++;
    }
  }
  if (n_pass == N_TESTS) {
    GLOG_PASS("Contiguous Reads : Passed ", n_pass, "/", N_TESTS, " tests");
  } else {
    GLOG_INFO("Contiguous Reads : Passed ", n_pass, "/", N_TESTS, " tests");
    GLOG_FAIL("Contiguous Reads : Failed ", N_TESTS - n_pass, "/", N_TESTS,
              " tests");
  }

  delete[] buf;
}

void test_write(FlashFileHandle& fhandle, FBLAS_UINT fsize,
                FBLAS_UINT max_buf_size) {
  FBLAS_UINT* buf = new FBLAS_UINT[max_buf_size / sizeof(FBLAS_UINT)];
  FBLAS_UINT* buf2 = new FBLAS_UINT[max_buf_size / sizeof(FBLAS_UINT)];

  FBLAS_UINT* backup_buf = new FBLAS_UINT[max_buf_size / sizeof(FBLAS_UINT)];
  FBLAS_UINT  max_read_offset = ROUND_DOWN(fsize - max_buf_size, 8);
  GLOG_ASSERT(max_read_offset > 8, "file size too small OR bad buffer size");

  FBLAS_UINT n_pass = 0;
  for (FBLAS_UINT i = 0; i < N_TESTS; i++) {
    FBLAS_UINT offset = ROUND_UP(rand() % max_read_offset, 8);
    FBLAS_UINT test_offset = ROUND_UP(rand() % max_read_offset, 8);
    FBLAS_UINT len = ROUND_UP(rand() % max_buf_size, 8);
    len = (len > 128 ? len : 128);
    memset(buf, 0, max_buf_size);
    memset(backup_buf, 0, max_buf_size);
    GLOG_INFO("Contiguous Write test #", i + 1, ": seed offset=", test_offset,
              ", offset=", offset, ", length=", len);
    fhandle.read(test_offset, len, buf, callback_fn);  // populate seed values
    GLOG_ASSERT(verify_iota(buf, test_offset, len),
                "contiguous read failed @ seed offset");
    fhandle.read(offset, len, backup_buf, callback_fn);  // store into backup
    GLOG_ASSERT(verify_iota(backup_buf, offset, len),
                "contiguous read failed @ offset");
    fhandle.write(offset, len, buf, callback_fn);  // write seed values
    memset(buf2, 0, len);
    fhandle.read(offset, len, buf2, callback_fn);  // read values
    // if (!verify_iota(buf, test_offset, len)) {
    if (memcmp(buf, buf2, len) != 0) {
      GLOG_FAIL("Contiguous Write test #", i + 1, " failed");
    } else {
      n_pass++;
    }
    fhandle.write(offset, len, backup_buf, callback_fn);  // restore from backup
  }
  if (n_pass == N_TESTS) {
    GLOG_PASS("Contiguous Writes : Passed ", n_pass, "/", N_TESTS, " tests");
  } else {
    GLOG_INFO("Contiguous Writes : Passed ", n_pass, "/", N_TESTS, " tests");
    GLOG_FAIL("Contiguous Writes : Failed ", N_TESTS - n_pass, "/", N_TESTS,
              " tests");
  }

  delete[] buf;
  delete[] buf2;
  delete[] backup_buf;
}

void test_sread(FlashFileHandle& fhandle, FBLAS_UINT fsize, StrideInfo sinfo) {
  FBLAS_UINT  buf_size = (sinfo.n_strides) * sinfo.len_per_stride;
  FBLAS_UINT* buf = new FBLAS_UINT[buf_size / sizeof(FBLAS_UINT)];
  FBLAS_UINT  max_read_offset = fsize - ((sinfo.n_strides) * sinfo.stride);
  GLOG_ASSERT(max_read_offset > 8, "file size too small OR bad stride info");

  FBLAS_UINT n_pass = 0;
  for (FBLAS_UINT i = 0; i < N_TESTS; i++) {
    FBLAS_UINT offset = ROUND_UP(rand() % max_read_offset, 8);
    StrideInfo cur_sinfo;
    cur_sinfo.n_strides = (rand() % sinfo.n_strides) + 1;
    cur_sinfo.len_per_stride = rand() % sinfo.len_per_stride;
    cur_sinfo.len_per_stride =
        ROUND_UP(cur_sinfo.len_per_stride, sizeof(FBLAS_UINT));
    cur_sinfo.len_per_stride =
        (cur_sinfo.len_per_stride > 0 ? cur_sinfo.len_per_stride : 128);
    cur_sinfo.stride = rand() % sinfo.stride;
    cur_sinfo.stride = ROUND_UP(cur_sinfo.stride, sizeof(FBLAS_UINT));
    if (cur_sinfo.len_per_stride > cur_sinfo.stride) {
      std::swap(cur_sinfo.len_per_stride, cur_sinfo.stride);
    }
    GLOG_INFO("Strided Read test#", i + 1, ": offset=", offset,
              ", len_per_stride=", cur_sinfo.len_per_stride,
              ", stride=", cur_sinfo.stride,
              ", n_strides=", cur_sinfo.n_strides);
    fhandle.sread(offset, cur_sinfo, buf, callback_fn);
    if (!verify_iota_stride(buf, offset, cur_sinfo)) {
      GLOG_FAIL("Strided Read test #", i + 1, " failed");
    } else {
      n_pass++;
    }
  }
  if (n_pass == N_TESTS) {
    GLOG_PASS("Strided Reads : Passed ", n_pass, "/", N_TESTS, " tests");
  } else {
    GLOG_INFO("Strided Reads : Passed ", n_pass, "/", N_TESTS, " tests");
    GLOG_FAIL("Strided Reads : Failed ", N_TESTS - n_pass, "/", N_TESTS,
              " tests");
  }

  delete[] buf;
}

void test_swrite(FlashFileHandle& fhandle, FBLAS_UINT fsize, StrideInfo sinfo) {
  FBLAS_UINT  buf_size = (sinfo.n_strides) * sinfo.len_per_stride;
  FBLAS_UINT* buf = new FBLAS_UINT[buf_size / sizeof(FBLAS_UINT)];
  FBLAS_UINT* backup_buf = new FBLAS_UINT[buf_size / sizeof(FBLAS_UINT)];
  FBLAS_UINT  max_read_offset = fsize - ((sinfo.n_strides) * sinfo.stride);
  GLOG_ASSERT(max_read_offset > 8, "file size too small OR bad stride info");

  FBLAS_UINT n_pass = 0;
  for (FBLAS_UINT i = 0; i < N_TESTS; i++) {
    FBLAS_UINT offset = ROUND_UP(rand() % max_read_offset, 8);
    FBLAS_UINT test_offset = ROUND_UP(rand() % max_read_offset, 8);
    StrideInfo cur_sinfo;
    cur_sinfo.n_strides = (rand() % sinfo.n_strides) + 1;
    cur_sinfo.len_per_stride = rand() % sinfo.len_per_stride;
    cur_sinfo.len_per_stride =
        ROUND_UP(cur_sinfo.len_per_stride, sizeof(FBLAS_UINT));
    cur_sinfo.len_per_stride =
        (cur_sinfo.len_per_stride > 0 ? cur_sinfo.len_per_stride : 128);
    cur_sinfo.stride = std::max(rand() % sinfo.stride, (FBLAS_UINT) 32);
    cur_sinfo.stride = ROUND_UP(cur_sinfo.stride, sizeof(FBLAS_UINT));
    cur_sinfo.stride = (cur_sinfo.stride > 0 ? cur_sinfo.stride : 128);
    if (cur_sinfo.len_per_stride > cur_sinfo.stride) {
      std::swap(cur_sinfo.len_per_stride, cur_sinfo.stride);
    }
    GLOG_INFO("Strided Write test #", i + 1, ": offset=", offset,
              ", len_per_stride=", cur_sinfo.len_per_stride,
              ", stride=", cur_sinfo.stride,
              ", n_strides=", cur_sinfo.n_strides,
              ". seed offset = ", test_offset);

    fhandle.sread(test_offset, cur_sinfo, buf,
                  callback_fn);  // populate seed values
    GLOG_ASSERT(verify_iota_stride(buf, test_offset, cur_sinfo),
                "strided read failed @ seed offset");
    fhandle.sread(offset, cur_sinfo, backup_buf,
                  callback_fn);  // store into backup
    GLOG_ASSERT(verify_iota_stride(backup_buf, offset, cur_sinfo),
                "strided read failed @ offset");
    fhandle.swrite(offset, cur_sinfo, buf, callback_fn);  // write seed values
    fhandle.sread(offset, cur_sinfo, buf, callback_fn);   // read values
    if (!verify_iota_stride(buf, test_offset, cur_sinfo)) {
      GLOG_FAIL("Strided Write test #", i + 1, " failed");
    } else {
      n_pass++;
    }
    fhandle.swrite(offset, cur_sinfo, backup_buf,
                   callback_fn);  // restore from backup
    usleep(10000);
  }
  if (n_pass == N_TESTS) {
    GLOG_PASS("Strided Writes : Passed ", n_pass, "/", N_TESTS, " tests");
  } else {
    GLOG_INFO("Strided Writes : Passed ", n_pass, "/", N_TESTS, " tests");
    GLOG_FAIL("Strided Writes : Failed ", N_TESTS - n_pass, "/", N_TESTS,
              " tests");
  }

  delete[] buf;
  delete[] backup_buf;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    GLOG_INFO(
        "usage : <exec> <temp_file_name> <temp_file_size (multiple of "
        "8, >= 16384)>");
    GLOG_FATAL("insufficient args: expected 2, got ", argc - 1);
  }

  std::string fname(argv[1]);

  FBLAS_UINT size = (FBLAS_UINT) std::stol(argv[2]);
  StrideInfo sinfo;
  sinfo.n_strides = MAX_SIMUL_REQS * 4;
  sinfo.len_per_stride = 512 * sizeof(FBLAS_UINT);
  sinfo.stride = 1024 * sizeof(FBLAS_UINT);
  FBLAS_UINT max_buf_size = sinfo.n_strides * sinfo.len_per_stride;
  if (size < (sinfo.n_strides + 2) * sinfo.stride) {
    size = (sinfo.n_strides + 2) * sinfo.stride;
    GLOG_WARN("Input file size too small - using size=", size);
  }

  // Create a file with the given size
  create_file(fname, size);

  // claim context for main-thread
  FlashFileHandle::register_thread();

  // open file handle
  FlashFileHandle fhandle;
  fhandle.open(fname, flash::Mode::READWRITE);

  // test file ops
  test_sread(fhandle, size, sinfo);
  test_write(fhandle, size, max_buf_size);
  test_swrite(fhandle, size, sinfo);
  test_read(fhandle, size, max_buf_size);

  // close file handle
  fhandle.close();

  // remove the file
  ::remove(fname.c_str());

  // give-back context for main-thread
  FlashFileHandle::deregister_thread();
}
