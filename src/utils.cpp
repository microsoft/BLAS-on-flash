// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bof_utils.h"

#include <unistd.h>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <vector>

namespace flash {
  void alloc_aligned(void** ptr, size_t size, size_t align) {
    *ptr = nullptr;
    GLOG_ASSERT(IS_ALIGNED(size), "invalid alloc_aligned call");
    *ptr = ::aligned_alloc(align, size);
    GLOG_ASSERT(*ptr != nullptr, "aligned_alloc failed");
    // GLOG_ASSERT(ret != EINVAL, "bad alignment value");
    // GLOG_ASSERT(ret != ENOMEM, "insufficient mem");
  }

  uint32_t fnv32a(const char* str, const uint32_t n_bytes) {
    uint32_t hval = 0x811c9dc5u;
    uint32_t fnv32_prime = 0x01000193u;
    uint32_t n_bytes_left = n_bytes;
    while (n_bytes_left--) {
      hval ^= (uint32_t) *str++;
      hval *= fnv32_prime;
    }

    return hval;
  }

  uint64_t fnv64a(const char* str, const uint64_t n_bytes) {
    const uint64_t fnv64Offset = 14695981039346656037u;
    const uint64_t fnv64Prime = 0x100000001b3u;
    uint64_t       hash = fnv64Offset;

    for (uint64_t i = 0; i < n_bytes; i++) {
      hash = hash ^ (uint64_t) *str++;
      hash *= fnv64Prime;
    }

    return hash;
  }

  // returns buffer size described by `sinfo`
  FBLAS_UINT buf_size(const StrideInfo sinfo) {
    return (sinfo.n_strides == 1)
               ? ROUND_UP(sinfo.len_per_stride, SECTOR_LEN) + SECTOR_LEN
               :  // overprovision
               sinfo.n_strides * sinfo.len_per_stride;
  };
}  // namespace flash
