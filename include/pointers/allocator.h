// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <sys/mman.h>
#include <cstdint>
#include <fstream>
#include <string>
#include "file_handles/flash_file_handle.h"
#include "logger.h"
#include "pointer.h"
#include "types.h"

namespace flash {

  // TODO: Make a custom allocator class w a user context interface

  template<typename T>
  flash_ptr<T> map_file(std::string fname, Mode mode, FBLAS_UINT foffset = 0,
                        int flags = 0) {
    GLOG_INFO("Mapping ", fname, ":", foffset, " to flash_ptr");

    flash_ptr<T> fptr;

    fptr.foffset = foffset;
    fptr.fop = new FlashFileHandle();
    fptr.fop->open(fname, mode);

    int prot;
    if (mode == Mode::READWRITE)
      prot = PROT_WRITE | PROT_READ;
    else
      prot = PROT_READ;

    void* ptr = mmap(
        nullptr, dynamic_cast<FlashFileHandle*>(fptr.fop)->file_sz - foffset,
        prot, MAP_SHARED | flags,
        dynamic_cast<FlashFileHandle*>(fptr.fop)->file_desc, 0);

    GLOG_ASSERT(ptr != MAP_FAILED, "mmap failed with error ", strerror(errno));
    fptr.ptr = (T*) ptr;

    return fptr;
  }

  template<typename T>
  void unmap_file(flash_ptr<T> fptr) {
    int ret = munmap(
        fptr.ptr,
        dynamic_cast<FlashFileHandle*>(fptr.fop)->file_sz - fptr.foffset);

    GLOG_ASSERT(ret != -1, "munmap failed with error ", strerror(errno));

    delete fptr.fop;
    // make `fptr` invalid
    fptr.fop = nullptr;
    fptr.foffset = (FBLAS_UINT) -1;
  }
}  // namespace flash
