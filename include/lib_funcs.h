// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <future>
#include "file_handles/flash_file_handle.h"
#include "pointers/allocator.h"
#include "pointers/pointer.h"
#include "scheduler/scheduler.h"
#include "utils.h"

namespace flash {
  extern std::string mnt_dir;
  // NOTE :: might give compilation issues if included twice
  // single includes
  extern Scheduler sched;
  extern Logger    __global_logger;

  // init `sched` and `__global_logger`
  void flash_setup(std::string mntdir);

  // teardown stuff setup in `flash_setup()`
  void flash_destroy();

  // TODO :: convert a normal pointer to flash pointer
  // construct MemFileHandle using `ptr` as base
  template<typename T>
  flash_ptr<T> make_flash_ptr(T *ptr, FBLAS_UINT n_bytes) {
    throw std::runtime_error("make_flash_ptr() not implemented");
    flash_ptr<T> fptr;
    return fptr;
  }

  // TODO :: convert a flash_ptr to normal pointer
  // Use `mmap` to map the file into memory
  template<typename T>
  T *make_ptr(flash_ptr<T> fptr) {
    throw std::runtime_error("make_ptr() not implemented");
    T *ptr;
    return ptr;
  }

  // memory related operations with flash_ptr
  // for small `n_bytes`, use this function
  template<typename T>
  void flash_memset(flash_ptr<T> fptr, int val, FBLAS_UINT n_bytes) {
    int *buf = new int[n_bytes / sizeof(int)];
    memset(buf, val, n_bytes);
    fptr.fop->write(fptr.foffset, n_bytes, buf, dummy_std_func);
  }

  template<typename T, typename W>
  void flash_memcpy(flash_ptr<T> dest, flash_ptr<W> &src, FBLAS_UINT n_bytes) {
    src.fop->copy(src.foffset, *dest.fop, dest.foffset, n_bytes,
                  dummy_std_func);
  }

  // sync ops on flash_ptr
  template<typename T>
  FBLAS_INT read_sync(T *dest, flash_ptr<T> src, size_t len) {
    return src.fop->read(src.foffset, len * sizeof(T), dest,
                         flash::dummy_std_func);
  }
  template<typename T>
  FBLAS_INT write_sync(flash_ptr<T> dest, T *src, size_t len) {
    return dest.fop->write(dest.foffset, len * sizeof(T), src,
                           flash::dummy_std_func);
  }

  // // async ops on flash_ptr
  // template<typename T>
  // std::future<FBLAS_INT> read_async(T *dest, flash_ptr<T> src, size_t len) {
  //   return std::async(std::launch::async, &FlashFileHandle::read, src.fop,
  //                     src.foffset, len * sizeof(T), dest,
  //                     flash::dummy_std_func);
  // }
  // template<typename T>
  // std::future<FBLAS_INT> write_async(flash_ptr<T> dest, T *src, size_t len) {
  //   return std::async(std::launch::async, &FlashFileHandle::write, dest.fop,
  //                     dest.foffset, len * sizeof(T), src,
  //                     flash::dummy_std_func);
  // }

  // truncates file backing `fptr` to `fptr.foffset + new_size` bytes
  template<typename T>
  void flash_truncate(flash_ptr<T> fptr, uint64_t new_size) {
    FlashFileHandle *ffh = dynamic_cast<FlashFileHandle *>(fptr.fop);
    GLOG_ASSERT(ffh != nullptr, "bad flash pointer");

    int res = ::ftruncate(ffh->file_desc, fptr.foffset + new_size);
    if (res != 0) {
      GLOG_ERROR("ftruncate failed with errno=", errno,
                 ", error=", ::strerror(errno));
    }
  }

  // malloc and free variants
  // Use flash mem allocator
  template<typename T>
  flash_ptr<T> flash_malloc(FBLAS_UINT n_bytes, std::string opt_name = "") {
    GLOG_ASSERT(n_bytes != 0, "cannot malloc 0 bytes");
    n_bytes = ROUND_UP(n_bytes, 4096);
    std::string fname = mnt_dir + std::string("tmp_");
    if (opt_name != "") {
      fname += opt_name + "_";
    }
    fname += std::to_string(n_bytes);
    FBLAS_INT fd = ::open(fname.c_str(), O_DIRECT | O_RDWR | O_CREAT, 00666);
    GLOG_ASSERT(fd != -1, "::open failed with errno=", errno);
    FBLAS_INT ret = ::ftruncate(fd, n_bytes);
    GLOG_ASSERT(ret != -1, "::ftruncate failed with errno=", errno);
    ret = ::close(fd);
    GLOG_ASSERT(ret != -1, "::close failed with errno=", errno);
    flash_ptr<T> fptr = map_file<T>(fname, Mode::READWRITE);
    return fptr;
  }

  template<typename T>
  void flash_free(flash_ptr<T> fptr) {
    unmap_file<T>(fptr);
    std::string fname = ((FlashFileHandle *) fptr.fop)->get_filename();
    GLOG_DEBUG("removing ", fname);
    ::remove(fname.c_str());
  }

}  // namespace flash
