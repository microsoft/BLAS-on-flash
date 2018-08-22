// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include "file_handles/file_handle.h"

namespace flash {
  class MemFileHandle : public BaseFileHandle {
    char *file_ptr;
    // mmap mapping info
    std::unordered_map<void *, FBLAS_UINT> mmaps_info;
    bool own;

   public:
    MemFileHandle();
    MemFileHandle(void *alloced_ptr, FBLAS_UINT size = 0);
    ~MemFileHandle();
    // Open & close ops
    FBLAS_INT open(std::string &fname, Mode fmode, FBLAS_UINT size = 0);
    FBLAS_INT close();
    // Contiguous read &write ops FBLAS_INT
    FBLAS_INT read(FBLAS_UINT offset, FBLAS_UINT len, void *buf,
                   const std::function<void(void)> &callback = dummy_std_func);
    FBLAS_INT write(FBLAS_UINT offset, FBLAS_UINT len, void *buf,
                    const std::function<void(void)> &callback = dummy_std_func);
    FBLAS_INT copy(FBLAS_UINT self_offset, BaseFileHandle &dest,
                   FBLAS_UINT dest_offset, FBLAS_UINT len,
                   const std::function<void(void)> &callback = dummy_std_func);

    // Non contiguous | strided read & write ops
    // NOTE :: stride >= len
    FBLAS_INT sread(FBLAS_UINT offset, StrideInfo sinfo, void *buf,
                    const std::function<void(void)> &callback = dummy_std_func);
    FBLAS_INT swrite(
        FBLAS_UINT offset, StrideInfo sinfo, void *buf,
        const std::function<void(void)> &callback = dummy_std_func);
    FBLAS_INT scopy(FBLAS_UINT self_offset, BaseFileHandle &dest,
                    FBLAS_UINT dest_offset, StrideInfo sinfo,
                    const std::function<void(void)> &callback = dummy_std_func);
  };
}  // namespace flash