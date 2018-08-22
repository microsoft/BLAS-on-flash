// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <functional>
#include <string>

#include "logger.h"
#include "types.h"

namespace flash {
  // FileHandle usage modes
  enum class Mode {
    READ,      // read-only mode
    WRITE,     // write-only mode
    READWRITE  // read & write mode
  };

  struct StrideInfo {
    FBLAS_UINT stride;
    FBLAS_UINT n_strides;
    FBLAS_UINT len_per_stride;

    operator std::string() const {
      return std::to_string(this->stride) + ":" +
             std::to_string(this->n_strides) + ":" +
             std::to_string(this->len_per_stride);
    }
    bool operator==(const StrideInfo &right) const {
      return this->stride == right.stride &&
             this->n_strides == right.n_strides &&
             this->len_per_stride == right.len_per_stride;
    }
  };

  extern std::function<void(void)> dummy_std_func;

  // File op interface
  class BaseFileHandle {
   public:
    virtual ~BaseFileHandle() {
    }
    // Open & close ops
    // Blocking calls
    virtual FBLAS_INT open(std::string &fname, Mode fmode,
                           FBLAS_UINT size = 0) = 0;
    virtual FBLAS_INT close() = 0;

    // Contiguous read & write ops
    virtual FBLAS_INT read(
        FBLAS_UINT offset, FBLAS_UINT len, void *buf,
        const std::function<void(void)> &callback = dummy_std_func) = 0;
    virtual FBLAS_INT write(
        FBLAS_UINT offset, FBLAS_UINT len, void *buf,
        const std::function<void(void)> &callback = dummy_std_func) = 0;
    virtual FBLAS_INT copy(
        FBLAS_UINT self_offset, BaseFileHandle &dest, FBLAS_UINT dest_offset,
        FBLAS_UINT                       len,
        const std::function<void(void)> &callback = dummy_std_func) = 0;

    // Non contiguous | strided read & write ops
    // NOTE :: stride >= len
    virtual FBLAS_INT sread(
        FBLAS_UINT offset, StrideInfo sinfo, void *buf,
        const std::function<void(void)> &callback = dummy_std_func) = 0;
    virtual FBLAS_INT swrite(
        FBLAS_UINT offset, StrideInfo sinfo, void *buf,
        const std::function<void(void)> &callback = dummy_std_func) = 0;
    virtual FBLAS_INT scopy(
        FBLAS_UINT self_offset, BaseFileHandle &dest, FBLAS_UINT dest_offset,
        StrideInfo                       sinfo,
        const std::function<void(void)> &callback = dummy_std_func) = 0;
  };
}  // namespace flash
