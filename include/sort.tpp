// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <unistd.h>
#include <numeric>
#include "scheduler/scheduler.h"
#include "tasks/sort_task.h"
#include "utils.h"

namespace flash {
  extern Scheduler sched;

  template<class T, class Comparator>
  FBLAS_INT sort(flash_ptr<T> in_fptr, flash_ptr<T> out_fptr, FBLAS_UINT n_vals,
           Comparator cmp) {
    FBLAS_UINT n_blks = std::ceil(std::sqrt(n_vals) / 1000);
    FBLAS_UINT blk_size = ROUND_UP(n_vals, n_blks) / n_blks;
    GLOG_INFO("Using ", n_blks, " blocks of size=", blk_size, " elements");
    FBLAS_UINT n_samples_per_blk = std::ceil(std::log10(n_vals));
    FBLAS_UINT n_samples = n_blks * n_samples_per_blk;
    T*   samples = new T[n_samples];
    FBLAS_UINT n_pivots = n_blks - 1;
    T*   pivots = new T[n_pivots];

    // split array and sort in-memory
    SampleSplit<T, Comparator>** split_tasks =
        new SampleSplit<T, Comparator>*[n_blks];
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      FBLAS_UINT arr_size = std::min(n_vals - i * blk_size, blk_size);
      split_tasks[i] = new SampleSplit<T, Comparator>(
          in_fptr, i * blk_size, arr_size, samples + (i * n_samples_per_blk),
          n_samples_per_blk, cmp);
    }
    // add dependency graph for split tasks
    for (FBLAS_UINT i = 0; i < n_blks; i += 2) {
      if (i == n_blks - 1) {
        // if only one task in total
        if (i == 0) {
          continue;
        } else {
          GLOG_DEBUG("adding dependency : ", i, "->", i - 1);
          split_tasks[i - 1]->add_parent(split_tasks[i]->get_id());
        }
      } else {
        if (i != 0) {
          GLOG_DEBUG("adding dependency : ", i, "->", i - 1);
          split_tasks[i - 1]->add_parent(split_tasks[i]->get_id());
        }
        GLOG_DEBUG("adding dependency : ", i, "->", i + 1);
        split_tasks[i + 1]->add_parent(split_tasks[i]->get_id());
      }
    }
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      sched.add_task(split_tasks[i]);
    }
    sleep_wait_for_complete(split_tasks, n_blks);
    GLOG_INFO("completed segment sorts");

    // compute pivot elements
    GLOG_INFO("generating pivots");
    std::sort(samples, samples + n_samples, cmp);
    std::random_shuffle(samples, samples + n_samples);
    memcpy(pivots, samples, n_pivots * sizeof(T));
    std::sort(pivots, pivots + n_pivots, cmp);
    GLOG_INFO("generated pivots");
    for (FBLAS_UINT i = 0; i < n_pivots; i++) {
      // GLOG_DEBUG("pivot:i=", i, ", piv=", pivots[i]);
    }

    // compute bucket boundaries
    GLOG_INFO("computing bucket boundaries");
    FBLAS_INT** starts = new FBLAS_INT*[n_blks];
    FBLAS_INT** ends = new FBLAS_INT*[n_blks];
    SampleSegment<T, Comparator>** segment_tasks =
        new SampleSegment<T, Comparator>*[n_blks];
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      starts[i] = new FBLAS_INT[n_pivots + 1];
      ends[i] = new FBLAS_INT[n_pivots + 1];
      segment_tasks[i] = new SampleSegment<T, Comparator>(
          starts[i], ends[i], pivots, n_pivots, in_fptr, i * blk_size,
          std::min(n_vals - i * blk_size, blk_size), cmp);
      sched.add_task(segment_tasks[i]);
    }
    sleep_wait_for_complete(segment_tasks, n_blks);
    GLOG_INFO("computed bucket boundaries");

    // compute prefix sums to get offsets for buckets in each block
    GLOG_INFO("merging buckets");
    FBLAS_UINT** sizes = new FBLAS_UINT*[n_blks];
    FBLAS_UINT** offsets = new FBLAS_UINT*[n_blks];
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      sizes[i] = new FBLAS_UINT[n_pivots + 1];
      offsets[i] = new FBLAS_UINT[n_pivots + 1];
      for (FBLAS_UINT j = 0; j <= n_pivots; j++) {
        if (starts[i][j] == -1 && ends[i][j] == -1) {
          sizes[i][j] = 0;
        } else {
          sizes[i][j] = (ends[i][j] - starts[i][j]) + 1;
          offsets[i][j] = (i * blk_size) + starts[i][j];
        }
      }
    }

    FBLAS_UINT* dest_sizes = new FBLAS_UINT[n_pivots + 1];
    FBLAS_UINT* dest_offsets = new FBLAS_UINT[n_pivots + 1];
    for (FBLAS_UINT i = 0; i <= n_pivots; i++) {
      dest_sizes[i] = 0;
      for (FBLAS_UINT j = 0; j < n_blks; j++) {
        dest_sizes[i] += sizes[j][i];
      }
    }
    // compute dest offsets
    dest_offsets[0] = 0;
    std::partial_sum(dest_sizes, dest_sizes + n_pivots, dest_offsets + 1);
    for (FBLAS_UINT i = 0; i <= n_pivots; i++) {
      GLOG_DEBUG("bucket:idx=", i, ", offset=", dest_offsets[i],
                 ", size=", dest_sizes[i]);
    }

    // compute bucket sizes
    std::vector<std::vector<FBLAS_UINT>> bsizes(n_pivots + 1, std::vector<FBLAS_UINT>());
    std::vector<std::vector<FBLAS_UINT>> boffsets(n_pivots + 1, std::vector<FBLAS_UINT>());
    // add only non-zero buckets
    for (FBLAS_UINT i = 0; i < n_pivots + 1; i++) {
      for (FBLAS_UINT j = 0; j < n_blks; j++) {
        if (sizes[j][i] > 0) {
          bsizes[i].push_back(sizes[j][i]);
          boffsets[i].push_back(offsets[j][i]);
        }
      }
    }

    // create and launch merge tasks
    SampleMerge<T, Comparator>** merge_tasks =
        new SampleMerge<T, Comparator>*[n_pivots + 1];
    for (FBLAS_UINT i = 0; i <= n_pivots; i++) {
      merge_tasks[i] = new SampleMerge<T, Comparator>(
          in_fptr, boffsets[i], bsizes[i], out_fptr, dest_offsets[i],
          dest_sizes[i], cmp);
    }
    for (FBLAS_UINT i = 0; i <= n_pivots; i += 2) {
      if (i == n_pivots) {
        // if only one task in total
        if (i == 0) {
          continue;
        } else {
          GLOG_DEBUG("adding dependency : ", i, "->", i - 1);
          merge_tasks[i - 1]->add_parent(merge_tasks[i]->get_id());
        }
      } else {
        if (i != 0) {
          GLOG_DEBUG("adding dependency : ", i, "->", i - 1);
          merge_tasks[i - 1]->add_parent(merge_tasks[i]->get_id());
        }
        GLOG_DEBUG("adding dependency : ", i, "->", i + 1);
        merge_tasks[i + 1]->add_parent(merge_tasks[i]->get_id());
      }
    }
    for (FBLAS_UINT i = 0; i <= n_pivots; i++) {
      sched.add_task(merge_tasks[i]);
    }
    sleep_wait_for_complete(merge_tasks, n_pivots + 1);
    GLOG_INFO("merged buckets");

    // remove split tasks
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      delete split_tasks[i];
    }
    delete[] split_tasks;

    delete[] samples;
    delete[] pivots;
    // remove starts and ends
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      delete[] starts[i];
      delete[] ends[i];
      delete[] sizes[i];
      delete[] offsets[i];
    }
    delete[] starts;
    delete[] ends;
    delete[] sizes;
    delete[] offsets;
    // remove segment tasks
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      delete segment_tasks[i];
    }
    delete[] segment_tasks;

    // cleanup dest offsets and sizes
    delete[] dest_offsets;
    delete[] dest_sizes;

    // cleanup merge tasks
    for (FBLAS_UINT i = 0; i <= n_pivots; i++) {
      delete merge_tasks[i];
    }
    delete[] merge_tasks;

    return 0;
  }
}  // namespace flash
