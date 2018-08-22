// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <string>
#include "file_handles/file_handle.h"
#include "file_handles/mem_file_handle.h"
#include "types.h"

namespace flash {

  template<typename T>
  struct flash_ptr {
    T*              ptr;
    FBLAS_UINT      foffset;  // offset from start of file
    BaseFileHandle* fop;      // associated file handle

    flash_ptr() : ptr(nullptr), foffset(0), fop(nullptr) {
    }
    flash_ptr(T* val) {
      GLOG_ASSERT(false, "Bad usage");
    }
    flash_ptr(T* val, FBLAS_UINT of, BaseFileHandle* bfh)
        : ptr(val), foffset(of), fop(bfh) {
    }

    flash_ptr operator+(FBLAS_UINT n_vals) const {
      return flash_ptr<T>(ptr + n_vals, foffset + n_vals * sizeof(T), fop);
    }

    template<typename X>
    bool operator==(const flash_ptr<X>& other) const {
      return ((void*) ptr == (void*) other.ptr) && (foffset == other.foffset) &&
             (fop == other.fop);
    }

    inline T* get_raw_ptr() const {
      return ptr;
    }

    // have to disable this if T == void
    template<class Q = T>
    typename std::enable_if<not std::is_same<Q, void>::value, T>::type&
    operator*() {
      return *ptr;
    }

    // type coercion
    template<typename W>
    operator flash_ptr<W>() const {
      return flash_ptr<W>((W*) ptr, foffset, fop);
    }

    operator std::string() const {
      return std::string("[") + std::to_string((uint64_t) this->fop) +
             std::string("-") + std::to_string(foffset) + std::string("]");
    }
  };

  class FlashPtrHasher {
   public:
    size_t operator()(flash_ptr<void> const& key) const {
      return std::hash<void*>()(key.get_raw_ptr());
    }
  };

  class FlashPtrEq {
   public:
    bool operator()(flash_ptr<void> const& t1,
                    flash_ptr<void> const& t2) const {
      return std::equal_to<void*>()(t1.get_raw_ptr(), t2.get_raw_ptr());
    }
  };

}  // namespace flash
