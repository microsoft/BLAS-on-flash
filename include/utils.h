// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <unistd.h>
#include <cstdlib>
#include <unordered_set>
#include "logger.h"
#include "tasks/task.h"

// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) \
  ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

// round down X to the nearest multiple of Y
#define ROUND_DOWN(X, Y) (((uint64_t)(X) / (Y)) * (Y))

// alignment tests
#define IS_512_ALIGNED(X) ((uint64_t)(X) % 512 == 0)
#define IS_4096_ALIGNED(X) ((uint64_t)(X) % 4096 == 0)

namespace flash {
  void alloc_aligned(void** ptr, size_t size, size_t align = SECTOR_LEN);

  template<class Task>
  void sleep_wait_for_complete(Task** tsks, FBLAS_UINT n_tasks,
                               FBLAS_UINT sleep_ms = 10) {
    FBLAS_UINT n_complete = 0;
    while (n_complete < n_tasks) {
      ::usleep(sleep_ms * 1000);  // 10ms
      n_complete = 0;
      for (FBLAS_UINT i = 0; i < n_tasks; i++) {
        if (tsks[i]->get_status() == Complete) {
          n_complete++;
        }
      }
    }
  }

  uint32_t fnv32a(const char* str, const uint32_t n_bytes);
  uint64_t fnv64a(const char* str, const uint64_t n_bytes);

  // returns buffer size described by `sinfo`
  FBLAS_UINT buf_size(const StrideInfo sinfo);

  // set intersection & differences
  template<typename T>
  inline std::unordered_set<T> set_intersection(
      const std::unordered_set<T>& first, const std::unordered_set<T>& second) {
    std::unordered_set<T> x;
    auto smaller = first.size() > second.size() ? second : first;
    auto larger = first.size() > second.size() ? first : second;
    for (auto el : smaller) {
      if (larger.find(el) != larger.end()) {
        x.insert(el);
      }
    }

    return x;
  }

  template<typename T>
  inline std::unordered_set<T> set_difference(
      const std::unordered_set<T> first, const std::unordered_set<T> second) {
    std::unordered_set<T> diff;
    for (auto v : first) {
      if (second.find(v) == second.end()) {
        diff.insert(v);
      }
    }

    return diff;
  }

  // return `buf` offset by `offset`
  template<typename T>
  inline T* offset_buf(T* buf, FBLAS_UINT offset) {
    return (T*) ((char*) buf + offset);
  }
}  // namespace flash
