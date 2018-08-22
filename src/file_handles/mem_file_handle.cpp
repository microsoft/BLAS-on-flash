// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "file_handles/mem_file_handle.h"
#include <sys/mman.h>
#include "types.h"
#include "utils.h"

flash::MemFileHandle::MemFileHandle() {
  this->file_ptr = nullptr;
}

flash::MemFileHandle::MemFileHandle(void* alloced_ptr, FBLAS_UINT size) {
  this->file_ptr = (char*) alloced_ptr;
  own = false;
}

flash::MemFileHandle::~MemFileHandle() {
  // take out pointers for mmap regions
  this->mmaps_info.clear();

  // free file mem
  if (own) {
    delete[] this->file_ptr;
  }
}

FBLAS_INT flash::MemFileHandle::open(std::string& fname, Mode fmode,
                                     FBLAS_UINT size) {
  if (size == 0) {
    GLOG_WARN("0 size");
  }
  // alloc mem for file
  this->file_ptr = new char[size];
  // zero file
  memset(file_ptr, 0, size);
  own = true;
  // Return success
  return 0;
}

FBLAS_INT flash::MemFileHandle::close() {
  return 0;
}

FBLAS_INT flash::MemFileHandle::read(
    FBLAS_UINT offset, FBLAS_UINT len, void* buf,
    const std::function<void(void)>& callback) {
  assert(this->file_ptr != nullptr);
  assert(buf != nullptr);
  assert(len != 0);
  memcpy(buf, this->file_ptr + offset, len);

  callback();

  return 0;
}

FBLAS_INT flash::MemFileHandle::write(
    FBLAS_UINT offset, FBLAS_UINT len, void* buf,
    const std::function<void(void)>& callback) {
  assert(this->file_ptr != nullptr);
  assert(buf != nullptr);
  assert(len != 0);
  memcpy(this->file_ptr + offset, buf, len);

  callback();

  return 0;
}

FBLAS_INT flash::MemFileHandle::copy(
    FBLAS_UINT self_offset, BaseFileHandle& dest, FBLAS_UINT dest_offset,
    FBLAS_UINT len, const std::function<void(void)>& callback) {
  assert(this->file_ptr != nullptr);
  assert(len != 0);
  // DRAM -> dest_file
  dest.write(dest_offset, len, this->file_ptr + self_offset);

  callback();

  return 0;
}

FBLAS_INT flash::MemFileHandle::sread(
    FBLAS_UINT offset, StrideInfo sinfo, void* buf,
    const std::function<void(void)>& callback) {
  assert(sinfo.len_per_stride != 0);
  assert(buf != nullptr);
  assert(this->file_ptr != nullptr);

  FBLAS_UINT n_reads = sinfo.n_strides;

  // issue reads
  char* start_ptr = this->file_ptr + offset;
  for (FBLAS_UINT idx = 0; idx < n_reads; idx++) {
    memcpy((char*) buf + (idx * sinfo.len_per_stride),
           start_ptr + (idx * sinfo.stride), sinfo.len_per_stride);
  }

  // return success
  return 0;
}

FBLAS_INT flash::MemFileHandle::swrite(
    FBLAS_UINT offset, StrideInfo sinfo, void* buf,
    const std::function<void(void)>& callback) {
  assert(sinfo.len_per_stride != 0);
  assert(buf != nullptr);
  assert(this->file_ptr != nullptr);

  FBLAS_UINT n_writes = sinfo.n_strides;

  // issue writes
  char* start_ptr = this->file_ptr + offset;
  for (FBLAS_UINT idx = 0; idx < n_writes; idx++) {
    memcpy(start_ptr + (idx * sinfo.stride),
           (char*) buf + (idx * sinfo.len_per_stride), sinfo.len_per_stride);
  }

  // return success
  return 0;
}

FBLAS_INT flash::MemFileHandle::scopy(
    FBLAS_UINT self_offset, BaseFileHandle& dest, FBLAS_UINT dest_offset,
    StrideInfo sinfo, const std::function<void(void)>& callback) {
  assert(sinfo.len_per_stride != 0);
  assert(this->file_ptr != nullptr);

  FBLAS_UINT n_copies = sinfo.n_strides;

  // copy data into one contiguous buffer
  char* buf = new char[n_copies * sinfo.len_per_stride];
  this->sread(self_offset, sinfo, buf, flash::dummy_std_func);

  // issue strided write to dest
  dest.swrite(dest_offset, sinfo, buf);

  // free mem
  delete[] buf;

  // return success
  return 0;
}
